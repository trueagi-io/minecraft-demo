from typing import List
from .mission_spec import MissionSpec
from .client_info import ClientInfo
from .mission_record_spec import MissionRecordSpec
from .world_state import WorldState


class AgentHost:
    def startMission(mission: MissionSpec, client_pool: List[ClientInfo], 
                     mission_record: MissionRecordSpec, role: int,
                     unique_experiment_id: str):
        self.testSchemasCompatible()
        if role < 0 or role >= mission.getNumberOfAgents():
            if mission.getNumberOfAgents() == 1:
                raise MissionException("Role " + str(role) + " is invalid for this single-agent mission - must be 0.",
                MissionException.MISSION_BAD_ROLE_REQUEST)
            else:
                raise MissionException("Role " + str(role) +\
                " is invalid for this \
                multi-agent mission - must be in \
                range 0-" + str(mission.getNumberOfAgents() - 1) + ".", MissionException.MISSION_BAD_ROLE_REQUEST)
        if mission.isVideoRequested(role):
            if mission.getVideoWidth( role ) % 4:
                raise MissionException("Video width must be divisible by 4.", MissionException.MISSION_BAD_VIDEO_REQUEST)
            if mission.getVideoHeight( role ) % 2:
                raise MissionException("Video height must be divisible by 2.", MissionException.MISSION_BAD_VIDEO_REQUEST)

        if self.world_state.is_mission_running:
            raise MissionException("A mission is already running.", MissionException.MISSION_ALREADY_RUNNING)
        pool = None 
        if role == 0:
            # We are the agent responsible for the integrated server.
            # If we are part of a multi-agent mission, our mission should have been started before any of the others are attempted.
            # This means we are in a position to reserve clients in the client pool:
            reservedAgents = asyncio.run(self.reserveClients(client_pool, mission.getNumberOfAgents()))

            if len(reservedAgents) != mission.getNumberOfAgents():
                # Not enough clients available - go no further.
                logging.error("Failed to reserve sufficient clients - throwing MissionException.")
                if (mission.getNumberOfAgents() == 1):
                    raise MissionException("Failed to find an available client for this mission - tried all the clients in the supplied client pool.", MissionException.MISSION_INSUFFICIENT_CLIENTS_AVAILABLE)
                else:
                    raise MissionException("There are not enough clients available in the ClientPool to start this " + str(mission.getNumberOfAgents()) + " agent mission.", MissionException.MISSION_INSUFFICIENT_CLIENTS_AVAILABLE)
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

    async def reserveClients(self, client_pool: List[ClientInfo]):
        reservedClients = set()
        # TODO - currently reserved for 20 seconds (the 20000 below) - make this configurable.
        request = "MALMO_REQUEST_CLIENT:" + MALMO_VERSION + ":20000:" +\
                    self.current_mission_init.getExperimentID() +"\n"
        # could be run concurenlty??
        for item in client_pool:
            logging.info("Sending reservation request to " + item.ip_address +\
                         ':' + item.control_port)
            try:
                fut = rpc.sendStringAndGetShortReply(item.ip_address, item.control_port, request)
                reply = await asyncio.wait_for(fut, timeout=3)
            except RuntimeError as e:
                logging.exception(e)
                continue
            logging.info("Reserving client, received reply from " + str(item.ip_address) + ": " + reply)
            malmo_reservation_prefix = "MALMOOK"
            malmo_mismatch = "MALMOERRORVERSIONMISMATCH"

            if reply.startswith(malmo_reservation_prefix):
                # Successfully reserved this client.
                reservedClients.add(item)
                clients_required -= 1
                if clients_required == 0:
                    break
            elif reply.startswith(malmo_mismatch):
                log.warning("Version mismatch - throwing MissionException.")
                raise MissionException( "Failed to find an available client for \
                                        this mission - tried all the clients in \
                                        the supplied client pool.", 
                                        MissionException.MISSION_VERSION_MISMATCH)            
            else:
                logging.error('unexpected reply ' + reply)
        #  Were there enough clients available?
        if clients_required > 0:
            for item in reservedClients:
                logging.info("Cancelling reservation request with " + item.ip_address + ":" +\
                              str(item.control_port))
                try:
                    fut = rpc.sendStringAndGetShortReply(self.io_service, item.ip_address, item.control_port, "MALMO_CANCEL_REQUEST\n", false)
                    reply = await asyncio.wait_for(fut, timeout=3)
                    logging.info("Cancelling reservation, received reply from " + str(item.ip_address) + ": " + reply)
                except RuntimeError as e:
                    # This is not expected, and probably means something bad has happened.
                    logging.error("Failed to cancel reservation request with " +\
                                   item.ip_address + ":" + str(item.control_port))
                    continue
                
        return reservedClients

    def findClient(self, client_pool: List[ClientInfo]):
        logging.info("Looking for client...")
        # As a reasonable optimisation, assume that clients are started in the order of their role, for multi-agent missions.
        # So start looking at position <role> within the client pool.
        # Eg, if the first four agents get clients 1,2,3 and 4 respectively, agent 5 doesn't need to waste time checking
        # the first four clients.
        num_clients = len(client_pool);
        for i in range(num_clients):
            item = client_pool.clients[(i + self.current_role) % num_clients]
            self.current_mission_init.setClientAddress( item.ip_address );
            self.current_mission_init.setClientMissionControlPort( item.control_port );
            self.current_mission_init.setClientCommandsPort( item.command_port );
            mission_init_xml = self.generateMissionInit() + "\n"
            logging.info("Sending MissionInit to " +\
                         item.ip_address, + ":" + item.control_port)
            
            try:
                reply = rpc.sendStringAndGetShortReply(item.ip_address, 
                                                       item.control_port, 
                                                       mission_init_xml)
            except RuntimeError as e:
                logging.info("No response from " + item.ip_address + ":" + item.control_port)

                # This is expected quite often - client is likely not running.
                continue
            logging.info("Looking for client, received reply from " + \
                         item.ip_address + ": " + reply)
            malmo_mission_accepted = "MALMOOK"
            if reply == malmo_mission_accepted:
                return # mission was accepted, now wait for the mission to start

        logging.warning("Failed to find an available client for this mission - throwing MissionException.")
        self.close()
        raise MissionException("Failed to find an available client for this mission -\
                                tried all the clients in the \
                                supplied client pool.", MissionException.MISSION_INSUFFICIENT_CLIENTS_AVAILABLE )


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
        self.current_mission_init = MissionInitSpec(mission, unique_experiment_id, role)
        self.current_mission_record = MissionRecord(mission_record)
        self.current_role = role
        if mission.isVideoRequested(self.current_role):
            self.video_server = listenForVideo(self.video_server,
                self.current_mission_init.getAgentVideoPort(),
                mission.getVideoWidth(self.current_role),
                mission.getVideoHeight(self.current_role),
                mission.getVideoChannels(self.current_role),
                TimestampedVideoFrame.VIDEO)


    def close(self):
        pass

    def generateMissionInit() -> str:
        prettyPrint = False
        generated_xml = self.current_mission_init.getAsXML(prettyPrint)
        return generated_xml

    def listenForVideo(self):
        raise NotImplementedError("unimplemented")
