import threading
import time
import copy
import logging
import asyncio
from typing import List
import xml.etree.ElementTree as ET

from .string_server import StringServer
from .timestamped_string import TimestampedString
from .timestamped_video_frame import FrameType, TimestampedVideoFrame
from .video_server import VideoServer
from .mission_spec import MissionSpec
from .mission_init_spec import MissionInitSpec
from .mission_record import MissionRecord
from .client_info import ClientInfo
from .mission_record_spec import MissionRecordSpec
from .world_state import WorldState
from .argument_parser import ArgumentParser
from .mission_exception import MissionException, MissionErrorCode
from .world_state_policy import VideoPolicy, RewardsPolicy, ObservationsPolicy 
from . import rpc
from .version import *
frim .consts import *




ET.register_namespace('', MALMO_NAMESPACE)
ET.register_namespace('xsi', XMLNS_XSI)

logger = logging.getLogger()


class AgentHost(ArgumentParser):
    def __init__(self):
        self.world_state_mutex = threading.Lock()
        self.io_service = asyncio.new_event_loop()
        self.th = threading.Thread(target=self.io_service.run_forever, daemon=True)
        self.th.start()
        self.video_server = None
        self.depth_server = None
        self.luminance_server = None
        self.world_state = WorldState()
        self.mission_control_server = None
        self.observations_server = None
        self.colourmap_server = None 
        self.commands_connection = None
        self.commands_stream = None
        self.rewards_server = None
        self.current_role = -1 
        self.video_policy = VideoPolicy.LATEST_FRAME_ONLY
        self.observations_policy = ObservationsPolicy.LATEST_OBSERVATION_ONLY

    def startMission(self, mission: MissionSpec, client_pool: List[ClientInfo],
                     mission_record: MissionRecordSpec, role: int,
                     unique_experiment_id: str):
        self.testSchemasCompatible()
        if role < 0 or role >= mission.getNumberOfAgents():
            if mission.getNumberOfAgents() == 1:
                raise MissionException("Role " + str(role) + " is invalid for self.single-agent mission - must be 0.",
                MissionErrorCode.MISSION_BAD_ROLE_REQUEST)
            else:
                raise MissionException("Role " + str(role) +\
                " is invalid for self.\
                multi-agent mission - must be in \
                range 0-" + str(mission.getNumberOfAgents() - 1) + ".", MissionErrorCode.MISSION_BAD_ROLE_REQUEST)
        if mission.isVideoRequested(role):
            if mission.getVideoWidth( role ) % 4:
                raise MissionException("Video width must be divisible by 4.", MissionErrorCode.MISSION_BAD_VIDEO_REQUEST)
            if mission.getVideoHeight( role ) % 2:
                raise MissionException("Video height must be divisible by 2.", MissionErrorCode.MISSION_BAD_VIDEO_REQUEST)

        if self.world_state.is_mission_running:
            raise MissionException("A mission is already running.", MissionErrorCode.MISSION_ALREADY_RUNNING)
        self.initializeOurServers( mission, mission_record, role, unique_experiment_id )

        pool = None
        if role == 0:
            # We are the agent responsible for the integrated server.
            # If we are part of a multi-agent mission, our mission should have been started before any of the others are attempted.
            # This means we are in a position to reserve clients in the client pool:
            reservedAgents = asyncio.run_coroutine_threadsafe(self.reserveClients(client_pool,
                                                                                  mission.getNumberOfAgents()),
                                                              self.io_service).result()

            if len(reservedAgents) != mission.getNumberOfAgents():
                # Not enough clients available - go no further.
                logger.error("Failed to reserve sufficient clients - throwing MissionException.")
                if (mission.getNumberOfAgents() == 1):
                    raise MissionException("Failed to find an available client for self.mission - tried all the clients in the supplied client pool.", MissionErrorCode.MISSION_INSUFFICIENT_CLIENTS_AVAILABLE)
                else:
                    raise MissionException("There are not enough clients available in the ClientPool to start self." + str(mission.getNumberOfAgents()) + " agent mission.", MissionErrorCode.MISSION_INSUFFICIENT_CLIENTS_AVAILABLE)
            pool = reservedAgents
        if( mission.getNumberOfAgents() > 1 and role > 0 \
            and not self.current_mission_init.hasMinecraftServerInformation()):
            raise NotImplementedError("role > 0 is not implemented yet")

        # work through the client pool until we find a client to run our mission for us
        self.findClient( pool )
        self.world_state.clear()
        if (self.current_mission_record.isRecording()):
            raise NotImplementedError('mission recoding is not implemented')


    def testSchemasCompatible(self):
        pass

    async def reserveClients(self, client_pool: List[ClientInfo], clients_required: int) -> set:
        reservedClients = set()
        # TODO - currently reserved for 20 seconds (the 20000 below) - make self.configurable.
        request = "MALMO_REQUEST_CLIENT:" + MALMO_VERSION + ":20000:" +\
                    self.current_mission_init.getExperimentID() +"\n"
        # could be run concurenlty??
        for item in client_pool:
            logger.info("Sending reservation request to " + str(item.ip_address) +\
                         ':' + str(item.control_port))
            try:
                fut = rpc.sendStringAndGetShortReply(item.ip_address, item.control_port, request)
                reply = await asyncio.wait_for(fut, timeout=3)
            except RuntimeError as e:
                logging.exception(e)
                continue
            logger.info("Reserving client, received reply from " + str(item.ip_address) + ": " + reply)
            malmo_reservation_prefix = "MALMOOK"
            malmo_mismatch = "MALMOERRORVERSIONMISMATCH"
            if reply.startswith(malmo_reservation_prefix):
                # Successfully reserved self.client.
                reservedClients.add(item)
                clients_required -= 1
                if clients_required == 0:
                    break
            elif reply.startswith(malmo_mismatch):
                log.warning("Version mismatch - throwing MissionException.")
                raise MissionException( "Failed to find an available client for \
                                        self.mission - tried all the clients in \
                                        the supplied client pool.",
                                        MissionErrorCode.MISSION_VERSION_MISMATCH)
            else:
                logger.error('unexpected reply ' + reply)
        #  Were there enough clients available?
        if clients_required > 0:
            for item in reservedClients:
                logger.info("Cancelling reservation request with " + item.ip_address + ":" +\
                              str(item.control_port))
                try:
                    fut = rpc.sendStringAndGetShortReply(self.io_service, item.ip_address, item.control_port, "MALMO_CANCEL_REQUEST\n", false)
                    reply = await asyncio.wait_for(fut, timeout=3)
                    logger.info("Cancelling reservation, received reply from " + str(item.ip_address) + ": " + reply)
                except RuntimeError as e:
                    # This is not expected, and probably means something bad has happened.
                    logger.error("Failed to cancel reservation request with " +\
                                   item.ip_address + ":" + str(item.control_port))
                    continue

        return reservedClients

    def findClient(self, client_pool: List[ClientInfo]):
        logger.info("Looking for client...")
        # As a reasonable optimisation, assume that clients are started in the order of their role, for multi-agent missions.
        # So start looking at position <role> within the client pool.
        # Eg, if the first four agents get clients 1,2,3 and 4 respectively, agent 5 doesn't need to waste time checking
        # the first four clients.
        pool = list(client_pool)
        num_clients = len(client_pool)
        for i in range(num_clients):
            item = pool[(i + self.current_role) % num_clients]
            self.current_mission_init.setClientAddress( item.ip_address )
            self.current_mission_init.setClientMissionControlPort( item.control_port )
            self.current_mission_init.setClientCommandsPort( item.command_port )
            mission_init_xml = self.generateMissionInit() + "\n"
            logger.info("Sending MissionInit to " + \
                         item.ip_address + ":" + str(item.control_port))

            try:
                reply = asyncio.run(rpc.sendStringAndGetShortReply(item.ip_address,
                                                       item.control_port,
                                                       mission_init_xml))
            except RuntimeError as e:
                logger.info("No response from " + item.ip_address + ":" + item.control_port)

                # This is expected quite often - client is likely not running.
                continue
            except asyncio.exceptions.IncompleteReadError as e:
                print(e)
                continue
            logger.info("Looking for client, received reply from " + \
                         item.ip_address + ": " + reply)
            malmo_mission_accepted = "MALMOOK"
            if reply == malmo_mission_accepted:
                return # mission was accepted, now wait for the mission to start

        logging.warning("Failed to find an available client for self.mission - throwing MissionException.")
        self.close()
        raise MissionException("Failed to find an available client for self.mission -\
                                tried all the clients in the \
                                supplied client pool.", MissionErrorCode.MISSION_INSUFFICIENT_CLIENTS_AVAILABLE )


    def peekWorldState(self) -> WorldState:
        with self.world_state_mutex:
            # Copy while holding lock.
            current_world_state = copy.deepcopy(self.world_state)
        return current_world_state

    def getWorldState(self) -> WorldState:
        with self.world_state_mutex:
            old_world_state = copy.deepcopy(self.world_state)
            self.world_state.clear()
            self.world_state.is_mission_running = old_world_state.is_mission_running
            self.world_state.has_mission_begun = old_world_state.has_mission_begun
            return old_world_state

    def getRecordingTemporaryDirectory(self) -> str:
        return current_mission_record.getTemporaryDirectory() if self.current_mission_record and self.current_mission_record.isRecording() else ""

    def initializeOurServers(self, mission: MissionSpec,
                             mission_record: MissionRecordSpec, role: int,
                             unique_experiment_id: str) -> None:
        logging.debug("Initialising servers...")
        # make a MissionInit structure with default settings
        self.current_mission_init = MissionInitSpec.from_param(mission, unique_experiment_id, role)
        self.current_mission_record = MissionRecord(mission_record)
        self.current_role = role
        self.listenForMissionControlMessages(self.current_mission_init.getAgentMissionControlPort())
        if mission.isVideoRequested(self.current_role):
            self.video_server = self.listenForVideo(self.video_server,
                self.current_mission_init.getAgentVideoPort(),
                mission.getVideoWidth(self.current_role),
                mission.getVideoHeight(self.current_role),
                mission.getVideoChannels(self.current_role),
                FrameType.VIDEO)
        if (mission.isDepthRequested(self.current_role)):
            self.depth_server = self.listenForVideo(self.depth_server,
                self.current_mission_init.getAgentDepthPort(),
                mission.getVideoWidth(self.current_role),
                mission.getVideoHeight(self.current_role),
                4,
                FrameType.DEPTH_MAP)
        if (mission.isLuminanceRequested(self.current_role)):
            self.luminance_server = self.listenForVideo(self.luminance_server,
                self.current_mission_init.getAgentLuminancePort(),
                mission.getVideoWidth(self.current_role),
                mission.getVideoHeight(self.current_role),
                1,
                FrameType.LUMINANCE)

        if (mission.isColourMapRequested(self.current_role)):
            self.colourmap_server = self.listenForVideo(self.colourmap_server,
                self.current_mission_init.getAgentColourMapPort(),
                mission.getVideoWidth(self.current_role),
                mission.getVideoHeight(self.current_role),
                3,
                FrameType.COLOUR_MAP)
        self.listenForRewards(self.current_mission_init.getAgentRewardsPort())
        self.listenForObservations(self.current_mission_init.getAgentObservationsPort())
        if self.commands_stream:
            self.commands_stream.close()
            self.commands_stream = None

        if self.current_mission_record.isRecordingCommands():
            self.commands_stream = open(self.current_mission_record.getCommandsPath(), 'wt')

        # if the requested port was zero then the system
        # will assign a free one, so store the resulting
        # value in the MissionInit node for sending to the client
        self.current_mission_init.setAgentMissionControlPort(self.mission_control_server.getPort())
        self.current_mission_init.setAgentObservationsPort(self.observations_server.getPort())
        if (self.video_server):
            self.current_mission_init.setAgentVideoPort(self.video_server.getPort())
        if (self.depth_server):
            self.current_mission_init.setAgentDepthPort(self.depth_server.getPort())
        if (self.luminance_server):
            self.current_mission_init.setAgentLuminancePort(self.luminance_server.getPort())
        if (self.colourmap_server):
            self.current_mission_init.setAgentColourMapPort(self.colourmap_server.getPort())
        self.current_mission_init.setAgentRewardsPort(self.rewards_server.getPort())


    def close(self):
        logger.debug("Closing AgentHost.")
        self.world_state.is_mission_running = False
        self.closeServers()
        self.closeRecording()

        # Ensure all TCP server ports are closed.

        if self.video_server:
            self.video_server.close()
            self.video_server = None 

        if self.depth_server:
            self.depth_server.close()
            self.depth_server = None 

        if self.luminance_server:
            self.luminance_server.close()
            self.luminance_server = None 

        if self.colourmap_server:
            self.colourmap_server->close ()
            self.colourmap_server = None 

        if self.observations_server:
            self.observations_server.close()
            self.observations_server = None 

        if self.rewards_server:
            self.rewards_server.close()
            self.rewards_server = None

        if self.mission_control_server:
            self.mission_control_server.close()
            self.mission_control_server = None
 
    def closeServers(self) -> None:
        if self.video_server:
            self.video_server.stopRecording()

        if self.depth_server:
            self.depth_server.stopRecording()
        

        if self.luminance_server:
            self.luminance_server.stopRecording()

        if self.colourmap_server:
            self.colourmap_server.stopRecording()
        

        if self.observations_server:
            self.observations_server.stopRecording()
        

        if self.rewards_server:
            self.rewards_server.stopRecording()
        

        if self.commands_stream.is_open():
            self.commands_stream.close()
        
        if self.commands_connection:
            self.commands_connection = None
        

    
    def generateMissionInit(self) -> str:
        prettyPrint = False
        generated_xml = self.current_mission_init.getAsXML(prettyPrint)
        return generated_xml

    def listenForVideo(self, video_server: VideoServer,
                       port: int, width: int, height: int,
                       channels: int, frametype: FrameType):
        path = None
        if frametype == FrameType.COLOUR_MAP:
            path = self.current_mission_record.getMP4ColourMapPath()
        elif frametype == FrameType.DEPTH_MAP:
            path = self.current_mission_record.getMP4DepthPath()
        elif frametype == FrameType.LUMINANCE:
            path = self.current_mission_record.getMP4LuminancePath()
        elif frametype == FrameType.VIDEO:
            pass
        else:
            path = self.current_mission_record.getMP4Path()

        if( video_server is None or
            (port != 0 and video_server.getPort() != port ) or
            video_server.getWidth() != width or
            video_server.getHeight() != height or
            video_server.getChannels() != channels or
            video_server.getFrameType() != frametype):

            if video_server is not None:
                video_server.close()

            # Can't use the server passed in - create a new one.
            ret_server = VideoServer(self.io_service, port, width, height, channels, frametype, self.onVideo)

            if (self.current_mission_record.isRecordingMP4(frametype)):
                ret_server.recordMP4(path,
                                     self.current_mission_record.getMP4FramesPerSecond(frametype),
                                     self.current_mission_record.getMP4BitRate(frametype),
                                     self.current_mission_record.isDroppingFrames(frametype))
            elif (self.current_mission_record.isRecordingBmps(frametype)):
                ret_server.recordBmps(self.current_mission_record.getTemporaryDirectory())

            ret_server.start()

        else:
            # re-use the existing video_server
            # but now we need to re-create the file writers with the new file names
            if (self.current_mission_record.isRecordingMP4(frametype)):
                video_server.recordMP4(path,
                                       self.current_mission_record.getMP4FramesPerSecond(frametype),
                                       self.current_mission_record.getMP4BitRate(frametype),
                                       self.current_mission_record.isDroppingFrames(frametype))
            elif (self.current_mission_record.isRecordingBmps(frametype)):
                video_server.recordBmps(self.current_mission_record.getTemporaryDirectory())
            ret_server = video_server

        ret_server.startRecording()
        return ret_server

    def sendCommand(self, command: str, key: str='') -> None:
        with self.world_state_mutex:
            if not self.commands_connection:
                error_message = TimestampedString(time.time(),
                                                  "AgentHost::sendCommand :\
                                                  commands connection is not open. Is the mission running?")
            self.world_state.errors.append(error_message)
            return
        full_command = command if not key else key + " " + command
        try:
            self.commands_connection.send(full_command)
        except RuntimeError as e:
            error_message = TimestampedString(time.time(),
                "AgentHost::sendCommand : failed to send command: " + str(e.args))
            self.world_state.errors.push_back(error_message)
            return

        if self.commands_stream:
            timestamp = datetime.datetime.now().isoformat()
            self.commands_stream.write(timestamp)
            self.commands_stream.write(" " + command + '\n')

    def openCommandsConnection(self) -> None:
        mod_commands_port = self.current_mission_init.getClientCommandsPort()
        if( mod_commands_port == 0 ):
            raise MissionException( "AgentHost::openCommandsConnection : client commands port \
                        is unknown! Has the mission started?", MissionErrorCode.MISSION_NO_COMMAND_PORT)

        mod_address = self.current_mission_init.getClientAddress()

        self.commands_connection = ClientConnection(self.io_service, mod_address, mod_commands_port )

    def listenForMissionControlMessages(self, port: int) -> int:
        if self.mission_control_server and ( port==0 or self.mission_control_server.getPort()==port ):
            return # can re-use existing server

        if self.mission_control_server is not None:
            self.mission_control_server.close()

        self.mission_control_server = StringServer(self.io_service, port, self.onMissionControlMessage, "mcp")
        self.mission_control_server.start()

    def onMissionControlMessage(self, xml: TimestampedString) -> None:
        try:
            logger.info('control %s', xml.text)
            elem = ET.fromstring(xml.text)
        except RuntimeError as e:
            print(e)
            text = "Error parsing mission control message as XML: " + repr(e) + ":\n" + xml.text[:200] + "...\n"
            error_message = TimestampedString(timestamp=xml.timestamp, text=text)
            with self.world_state_mutex:
                self.world_state.errors.append(error_message)
            return
        if not elem:
            text = "Empty XML string in mission control message"
            error_message = TimestampedString(timestamp=xml.timestamp, text=text)
            with self.world_state_mutex:
                self.world_state.errors.append(error_message)
            return
        root_node_name = elem.tag
        with self.world_state_mutex:
            if( not world_state.is_mission_running and root_node_name == "MissionInit" ):
                validate = True
                self.current_mission_init = MissionInitSpec.fromstr(xml.text, validate)
                self.world_state.is_mission_running = True
                self.world_state.has_mission_begun = True
                self.openCommandsConnection()
            elif root_node_name == "MissionEnded":
                try:
                    mission_ended = MissionEndedXML(xml.text)
                    status = mission_ended.getStatus()
                    if status != MissionEndedXML.ENDED and status != MissionEndedXML.PLAYER_DIED:
                        text = "Mission ended abnormally: " + mission_ended.getHumanReadableStatus()
                        error_message = TimestampedString(timestamp=xml.timestamp, text=text)
                        self.world_state.errors.append(error_message)
                    if self.world_state.is_mission_running:
                        reward = mission_ended.getReward()
                        if reward.size() != 0:
                            final_reward = TimestampedReward(timestamp=xml.timestamp, reward=reward)
                            self.processReceivedReward(final_reward)
                            self.rewards_server.recordMessage(TimestampedString(xml.timestamp, final_reward.getAsSimpleString()))

                    # Close our servers now, before we finish writing the MissionEnded message.
                    self.closeServers()

                    # Add some diagnostics of our own before this gets to the agent:
                    if self.video_server or self.luminance_server or self.depth_server or self.colourmap_server:
                        for vd in mission_ended.videoDataAttributes():
                            vs: VideoServer = None
                            if (vd.frame_type == "VIDEO"):
                                vs = self.video_server
                            elif (vd.frame_type == "DEPTH_MAP"):
                                vs = self.depth_server;
                            elif (vd.frame_type == "LUMINANCE")
                                vs = self.luminance_server;
                            elif (vd.frame_type == "COLOUR_MAP")
                                vs = self.colourmap_server;
                            if vs: 
                                vd.frames_received =  vs.receivedFrames()
                                vd.frames_written = vs.writtenFrames()
                    xml.text = mission_ended.toXml()
                except RuntimeError as e:
                    
    def onVideo(self, message: TimestampedVideoFrame) -> None:
        with self.world_state_mutex:
            if self.video_policy == VideoPolicy.LATEST_FRAME_ONLY:
                if (message.frametype == FrameType.COLOUR_MAP): 
                    self.world_state.video_frames_colourmap.clear()
                else:
                    self.world_state.video_frames.clear()
            else:
                print('append')
            if message.frametype == FrameType.COLOUR_MAP:
                self.world_state.video_frames_colourmap.append(message)
            else:
                self.world_state.video_frames.append(message)
            self.world_state.number_of_video_frames_since_last_state += 1

    def listenForRewards(self, port: int) -> None:
        if not self.rewards_server or ( port != 0 and self.rewards_server.getPort() != port ):
            if (self.rewards_server is not None):
                rewards_server.close()

            self.rewards_server = StringServer(self.io_service, port, self.onReward, "rew")
            self.rewards_server.start()

        if (self.current_mission_record.isRecordingRewards()):
            self.rewards_server.record(self.current_mission_record.getRewardsPath())

    def onReward(self, message: TimestampedString) -> None:
        with self.world_state_mutex:
            try:
                reward = TimestampedReward() 
                reward.createFromSimpleString(message.timestamp, message.text)
                self.processReceivedReward(reward)
            except RuntimeError as e: 
                text = "Error parsing Reward message: " + message.text
                logger.error(text)
                logger.exception(e)
                error_message = TimestampedString(message.timestamp, text=text)
                self.world_state.errors.append(error_message)

    def listenForObservations(self, port: int) -> None:
        if not self.observations_server or ( port != 0 and self.observations_server.getPort() != port ):
            if (self.observations_server is not None):
                self.observations_server.close()
            
            self.observations_server = StringServer(self.io_service, port, self.onObservation, "obs")
            self.observations_server.start()
        
        if self.current_mission_record.isRecordingObservations():
            self.observations_server.record(self.current_mission_record.getObservationsPath())

    def onObservation(self, message: TimestampedString) -> None:
        with self.world_state_mutex:
            if self.observations_policy == ObservationsPolicy.LATEST_OBSERVATION_ONLY:
                self.world_state.observations.clear()
                self.world_state.observations.append(message)
            elif self.observations_policy == ObservationsPolicy.KEEP_ALL_OBSERVATIONS:
                print('append')
                self.world_state.observations.append(message)
            else:
                raise RuntimeError('unexpected observation policy ' + str(self.self.observations_policy))

            self.world_state.number_of_observations_since_last_state += 1

    def closeRecording(self):
        pass

    def __del__(self):
        self.io_service.stop()
