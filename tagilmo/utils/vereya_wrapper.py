import json
import uuid
import time
import math
import sys
import concurrent.futures
import threading
import logging
import re
import os
import errno
from collections import defaultdict
from typing import Optional, Any

from tagilmo import VereyaPython as VP

import numpy
import tagilmo.utils.mission_builder as mb
from tagilmo.utils.mathutils import *
from tagilmo.VereyaPython import TimestampedString, TimestampedVideoFrame, FrameType

logger = logging.getLogger('vereya')


module = VP


def get_recordings_directory(agent_host):
    # Check the dir passed in:
    recordingsDirectory = agent_host.getStringArgument('recording_dir')
    if recordingsDirectory:
        # If we're running as an integration test, we want to send all our recordings
        # to the central test location specified in the environment variable MALMO_TEST_RECORDINGS_PATH:
        if agent_host.receivedArgument("test"):
            try:
                test_path = os.environ['MALMO_TEST_RECORDINGS_PATH']
                if test_path:
                    recordingsDirectory = os.path.join(test_path, recordingsDirectory)
            except:
                pass
        # Now attempt to create the folder we want to write to:
        try:
            os.makedirs(recordingsDirectory)
        except OSError as exception:
            if exception.errno != errno.EEXIST: # ignore error if already existed
                raise
    return recordingsDirectory


class MCConnector:

    @staticmethod
    def yawDelta(yawRad):
        return [-math.sin(yawRad), 0, math.cos(yawRad)]

    def setMissionXML(self, missionXML, module=VP):
        self.missionDesc = missionXML
        self.mission = module.MissionSpec(missionXML.xml(), True)
        self.mission_record = module.MissionRecordSpec()

    def __init__(self, missionXML, serverIp='127.0.0.1'):
        self.missionDesc = None
        self.mission = None
        self.mission_record = None
        self.prev_mobs = defaultdict(set) # host -> set mapping
        self.agentId = 0
        self._data_lock = threading.RLock()
        self.setUp(VP, missionXML, serverIp=serverIp)

    def setUp(self, module, missionXML, serverIp='127.0.0.1'):
        self.serverIp = serverIp
        self.setMissionXML(missionXML, module)
        agentIds = len(missionXML.agentSections)
        self.agent_hosts = dict()
        self.agent_hosts.update({n: module.AgentHost() for n in range(agentIds)})
        self.agent_hosts[0].parse( sys.argv )
        if self.receivedArgument('recording_dir'):
            recordingsDirectory = get_recordings_directory(self.agent_hosts[0])
            self.mission_record.recordRewards()
            self.mission_record.recordObservations()
            self.mission_record.recordCommands()
            self.mission_record.setDestination(recordingsDirectory + "//" + "lastRecording.tgz")
            if self.agent_hosts[0].receivedArgument("record_video"):
                self.mission_record.recordMP4(24, 2000000)
        self.client_pool = module.ClientPool()
        for x in range(10000, 10000 + agentIds):
            self.client_pool.add( module.ClientInfo(serverIp, x) )
        self.worldStates = [None] * agentIds
        self.observe = {k: None for k in range(agentIds)}
        self.isAlive = [True] * agentIds
        self.frames = dict({n: None for n in range(agentIds)})
        self.segmentation_frames = dict({n: None for n in range(agentIds)})
        self._last_obs = dict() # agent_host -> TimestampedString


    def getVersion(self, num=0) -> str:
        return self.agent_hosts[num].version

    def receivedArgument(self, arg):
        return self.agent_hosts[0].receivedArgument(arg)

    def safeStart(self):
        # starting missions
        expId = str(uuid.uuid4()) # will not work for multithreading, distributed agents, etc. (should be same for all agents to join the same server/mission)
        for role in range(len(self.agent_hosts)):
            used_attempts = 0
            max_attempts = 5
            while True:
                try:
                    # Attempt start:
                    self.agent_hosts[role].startMission(self.mission, self.client_pool, self.mission_record, role, expId)
                    #self.agent_hosts[role].startMission(self.mission, self.mission_record)
                    break
                except (VP.MissionException) as e:
                    errorCode = e.details.errorCode
                    if errorCode == VP.MissionErrorCode.MISSION_SERVER_WARMING_UP:
                        print("Server not quite ready yet - waiting...")
                        time.sleep(2)
                    elif errorCode == VP.MissionErrorCode.MISSION_INSUFFICIENT_CLIENTS_AVAILABLE:
                        print("Not enough available Minecraft instances running.")
                        used_attempts += 1
                        if used_attempts < max_attempts:
                            print("Will wait in case they are starting up.", max_attempts - used_attempts, "attempts left.")
                            time.sleep(2)
                    elif errorCode == VP.MissionErrorCode.MISSION_SERVER_NOT_FOUND:
                        print("Server not found - has the mission with role 0 been started yet?")
                        used_attempts += 1
                        if used_attempts < max_attempts:
                            print("Will wait and retry.", max_attempts - used_attempts, "attempts left.")
                            time.sleep(2)
                    elif errorCode == VP.MissionErrorCode.MISSION_ALREADY_RUNNING:
                        print('The mission already running')
                        return True
                    elif errorCode == VP.MissionErrorCode.MISSION_VERSION_MISMATCH:
                        print("wrong Malmo version")
                        global module
                        desc = self.missionDesc
                        # clean worldgen
                        worldgen = self.missionDesc.serverSection.handlers.worldgenerator
                        match = re.match('.*(forceReuse="\w+").*', worldgen)
                        if match:
                            desc.serverSection.handlers.worldgenerator = worldgen.replace(match.group(1), "")
                        if module is VP:
                            module = MP
                            desc.namespace = "ProjectMalmo.microsoft.com"
                        else:
                            module = VP
                            desc.namespace = "ProjectMalmo.singularitynet.io"
                        self.setUp(module, desc, self.serverIp)
                        continue
                    else:
                        print("Other error:", e)
                        return False
                except Exception as e:
                    print(e)
                    return False
                if used_attempts == max_attempts:
                    print("All attempts to start the mission are failed.")
                    return False
        # waiting for the real start
        start_flags = [False for a in self.agent_hosts]
        start_time = time.time()
        time_out = 620  # Allow a two minute timeout.
        while not all(start_flags) and time.time() - start_time < time_out:
            states = [a.peekWorldState() for a in self.agent_hosts.values()]
            start_flags = [w.has_mission_begun for w in states]
            errors = [e for w in states for e in w.errors]
            if len(errors) > 0:
                print("Errors waiting for mission start:")
                for e in errors:
                    print(e.text)
                    print("Quiting.")
                    return False
                time.sleep(0.1)
                print(".", end=' ')
            if time.time() - start_time >= time_out:
                print("Timed out while waiting for mission to start - quiting.")
                return False
        return True

    @staticmethod
    def connect(name=None, video=False, seed=None):
        if video:
            video_producer = mb.VideoProducer(want_depth=False)
            agent_handlers = mb.AgentHandlers(video_producer=video_producer)
        else:
            agent_handlers = mb.AgentHandlers()
        if name is not None:
            agent_section = mb.AgentSection(name=name,
                agenthandlers=agent_handlers)
        else:
            agent_section = mb.AgentSection(agenthandlers=agent_handlers)
        miss = mb.MissionXML(agentSections=[agent_section])
        miss.serverSection.initial_conditions.allowedmobs = "Pig Sheep Cow Chicken Ozelot Rabbit Villager"
        world = mb.defaultworld(forceReset="false", forceReuse="true", seed=seed)
        miss.setWorld(world)
        mc = MCConnector(miss)
        mc.safeStart()
        return mc

    def is_mission_running(self, agentId=0):
        world_state = self.agent_hosts[agentId].getWorldState()
        return world_state.is_mission_running

    def sendCommand(self, command, agentId=None):
        if agentId is None:
            agentId = self.agentId
        if agentId not in self.agent_hosts:
            logger.error(f"can't send command to {agentId}, it's not in agent_hosts")
            return
        self.agent_hosts[agentId].sendCommand(command)

    def observeProc(self, agentId=None):
        r = range(len(self.agent_hosts)) if agentId is None else range(agentId, agentId+1)
        for n in r:
            self.worldStates[n] = self.agent_hosts[n].getWorldState()
            self.isAlive[n] = self.worldStates[n].is_mission_running
            obs = self.worldStates[n].observations
            if obs:
                self.updateObservations(obs[-1], n)
            else:
                self.updateObservations(None, n)
            # might need to wait for a new frame
            frames = self.worldStates[n].video_frames
            segments = self.worldStates[n].video_frames_colourmap if self.supportsSegmentation() else None
            if frames:
                self.updateFrame(frames[0], n)
            else:
                self.updateFrame(None, n)
            if segments:
                self.updateSegmentation(segments[0], n)
            else:
                self.updateSegmentation(None, n)

    def updateFrame(self, frame: TimestampedVideoFrame, n: int) -> None:
        self.frames[n] = frame

    def updateSegmentation(self, segmentation_frame: TimestampedVideoFrame, n: int) -> None:
        self.segmentation_frames[n] = segmentation_frame

    def updateObservations(self, obs: Optional[TimestampedString], n: Any) -> None:
        if obs is None:
            self.observe[n] = None
            return
        agent_host = self.agent_hosts.get(n, None)
        if agent_host is None or obs == self._last_obs.get(agent_host, None):
            return

        data = json.loads(obs.text)

        with self._data_lock:
            self.observe[n] = data
            self._process_mobs(data, self.agent_hosts[n])
            self._last_obs[agent_host] = obs

    def _process_mobs(self, data, host):
        mobs = set()
        for key, value in data.get('ControlledMobs', dict()).items():
            logger.info(f'adding mob {key}')
            self.observe[key] = value
            self.agent_hosts[key] = host
            if key not in self.segmentation_frames:
                self.segmentation_frames[key] = None
                self.frames[key] = None
            mobs.add(key)
        missing = self.prev_mobs[host] - mobs
        for m in missing:
            logger.info(f"removing mob {m}")
            self.observe.pop(m)
            self.agent_hosts.pop(m)
            self.frames.pop(m)
            self.segmentation_frames.pop(m)
            self.frames.pop(m)
        self.prev_mobs[host] = mobs
        self._all_mobs = set().union(*self.prev_mobs.values())

    def getImageFrame(self, agentId=0):
        return self.frames[agentId]

    def getSegmentationFrame(self, agentId=0):
        return self.segmentation_frames[agentId]

    def getImage(self, agentId=0):
        if self.frames[agentId] is not None:
            return numpy.frombuffer(self.frames[agentId].pixels, dtype=numpy.uint8)
        return None

    def getSegmentation(self, agentId=0):
        if self.segmentation_frames[agentId] is not None:
            return numpy.frombuffer(self.segmentation_frames[agentId].pixels, dtype=numpy.uint8)
        return None

    def getAgentPos(self, agentId=None):
        if agentId is None: agentId = self.agentId
        data = self.observe.get(agentId, None)
        if (data is not None) and ('XPos' in data):
            return [data[key] for key in ['XPos', 'YPos', 'ZPos', 'Pitch', 'Yaw']]
        return None

    def getControlledMobs(self, agentId=None):
        return self.getParticularObservation('ControlledMobs', agentId)

    def getFullStat(self, key, agentId=None):
        """
            Dsc: this function was intended to return observations from full stats.
                 However, full stats don't have a dedicated key, but are placed directly
                 (each stat has its own key). Thus, this procedure can return any observation.
            FullStat:
                keys (position): 'XPos', 'YPos', 'ZPos', 'Pitch', 'Yaw'
                keys (status)  : 'Life', 'Food', 'Air', 'IsAlive'
                keys (more)    : 'XP', 'Score', 'Name', 'WorldTime', 'TotalTime', 'DistanceTravelled', 'TimeAlive',
                                 'MobsKilled', 'PlayersKilled', 'DamageTaken', 'DamageDealt'
                keys (new)     : 'input_type', 'isPaused'
        """

        return self.getParticularObservation(key, agentId)

    def getLineOfSights(self, agentId=None):
        data = self.getParticularObservation('LineOfSight', agentId)
        if data is None:
            return None
        los = data.get('LineOfSight', None)
        if los is not None and los['hitType'] != 'MISS':
            los['type'] = los['type'].replace("minecraft:", "")
            return los
        return None

    def getLineOfSight(self, key, agentId=None):
        # keys: 'hitType', 'x', 'y', 'z', 'type', 'prop_snowy', 'inRange', 'distance'
        los = self.getLineOfSights(agentId)
        return los[key] if los is not None and key in los else None

    def getNearEntities(self, agentId=None):
        data = self.getParticularObservation('ents_near' , agentId)
        if data is None:
            return

        for e in data:
            if 'name' in e:
                name = e['name']
                e['name'] = name.lower().replace(' ', '_')
        return data

    def getNearPickableEntities(self, agentId=None):
        entities = self.getNearEntities(agentId)
        if entities is None:
            return None
        pickableEntities = []
        for ent in entities:
            if ent['type'] == 'item':
                pickableEntities.append(ent)
        return pickableEntities

    def getNearGrid(self, agentId=None):
        return self.getParticularObservation('grid_near', agentId)

    def getLife(self, agentId=None):
        return self.getParticularObservation('Life', agentId)

    def getAir(self, agentId=None):
        return self.getParticularObservation('Air', agentId)

    def getChat(self, agentId=None):
        return self.getParticularObservation('Chat', agentId)

    def getParticularObservation(self, observation_name, agentId=None):
        if agentId is None: agentId = self.agentId
        obs = self.observe.get(agentId, None)
        if obs is not None:
            return obs.get(observation_name, None)

    def getItemList(self, agentId=None):
        return self.getParticularObservation('item_list', agentId)

    def getBlocksDropsList(self, agentId=None):
        return self.getParticularObservation('block_item_tool_triple', agentId)

    def getBlockFromBigGrid(self, agentId=None):
        return self.getParticularObservation('block_pos_big_grid', agentId)

    def getNonSolidBlocks(self, agentId=None):
        return self.getParticularObservation('nonsolid_blocks', agentId)

    def getRecipeList(self, agentId=None):
        return self.getParticularObservation('recipes', agentId)

    def getInventory(self, agentId=None):
        return self.getParticularObservation('inventory', agentId)

    def getGridBox(self, agentId=None):
        return self.missionDesc.agentSections[agentId].agenthandlers.observations.gridNear

    def gridIndexToPos(self, index, agentId=None):
        gridBox = self.getGridBox(agentId)
        gridSz = [gridBox[i][1]-gridBox[i][0]+1 for i in range(3)]
        y = index // (gridSz[0] * gridSz[2])
        index -= y * (gridSz[0] * gridSz[2])
        y += gridBox[1][0]
        z = index // gridSz[0] + gridBox[2][0]
        x = index % gridSz[0] + gridBox[0][0]
        return [x, y, z]

    def dirToPos(self, aPos, pos):
        '''
        Get pitch and yaw between two arbitrary positions
        (agent height is not taken into account)
        '''
        dx = pos[0] - aPos[0]
        dz = pos[2] - aPos[2]
        yaw = -math.atan2(dx, dz)
        pitch = -math.atan2(pos[1] - aPos[1], math.sqrt(dx * dx + dz * dz))
        return [pitch, yaw]

    def supportsVideo(self):
        return self.missionDesc.hasVideo()

    def supportsSegmentation(self):
        return self.missionDesc.hasSegmentation()

    def stop(self, idx=0):
        if idx < len(self.agent_hosts):
            self.agent_hosts[idx].stop()

    def getHumanInputs(self, agentId=None):
        return self.getParticularObservation('input_events', agentId)

    def placeBlock(self, x: int, y: int, z: int, block_name: str, placement: str, agentId=0):
        self.agent_hosts[agentId].sendCommand("placeBlock {} {} {} {} {}".format(x, y, z, block_name, placement))

    def _sendMotionCommand(self, command, value, agentId=None):
        if agentId is None: agentId = self.agentId
        if agentId in self._all_mobs:
            self.sendCommand(f'{command} {agentId} {value}', agentId)
        else:
            self.sendCommand(f'{command} {value}', agentId)

    def strafe(self, value, agentId=None):
        return self._sendMotionCommand('strafe', value, agentId)

    def move(self, value, agentId=None):
        return self._sendMotionCommand('move', value, agentId)

    def jump(self, value, agentId=None):
        return self._sendMotionCommand('jump', value, agentId)

    def pitch(self, value, agentId=None):
        return self._sendMotionCommand('pitch', value, agentId)

    def turn(self, value, agentId=None):
        return self._sendMotionCommand('turn', value, agentId)


class RobustObserver:

    deadlyBlocks = ['lava', 'cactus']
    # Should we merge these types of commands in one list?
    explicitlyPoseChangingCommands = ['move', 'jump', 'pitch', 'turn']
    implicitlyPoseChangingCommands = ['attack']

    def __init__(self, mc, agentId = 0):
        self.mc = mc
        self.passableBlocks = []
        self.agentId = agentId
        self.tick = 0.02
        self.methods = ['getNearEntities', 'getNearGrid', 'getAgentPos', 'getLineOfSights', 'getLife',
                        'getAir', 'getInventory', 'getImageFrame', 'getSegmentationFrame', 'getChat', 'getRecipeList',
                        'getItemList', 'getHumanInputs', 'getNearPickableEntities', 'getBlocksDropsList',
                        'getNonSolidBlocks', 'getBlockFromBigGrid',
                        'getControlledMobs']

        self.canBeNone = ['getLineOfSights', 'getChat', 'getHumanInputs', 'getItemList', 'getRecipeList',
                          'getNearPickableEntities', 'getBlocksDropsList', 'getNonSolidBlocks', 'getBlockFromBigGrid',
                          'getControlledMobs']

        self.events = ['getChat', 'getHumanInputs', 'getBlockFromBigGrid']

        self.readEvents = {event : False for event in self.events}

        if not self.mc.supportsVideo():
            self.canBeNone.append('getImageFrame')
        if not self.mc.supportsSegmentation():
            self.canBeNone.append('getSegmentationFrame')
        self.max_dt = 1.0
        self.cached = {method : (None, 0) for method in self.methods}
        for event in self.events:
            self.cached[event] = [(None, 0)]
        self.cbuff_history_len = 10
        self.cached_buffer = {method: (None, 0) for method in self.methods}
        self.cached_buffer_list = [self.cached_buffer]
        self.commandBuffer = []
        self.expectedCommandsBuffer = []
        self.thread = None
        self.lock = threading.RLock()
        self._time_sleep = 0.05
        self.mc.agent_hosts[self.agentId].addOnObservationCallback(self.onObservationChanged)
        self.mc.agent_hosts[self.agentId].addOnNewFrameCallback(self.onNewFrameCallback)

    def updatePassableBlocks(self):
        nonsolidblocks = self.__getNonSolidBlocks()
        self.passableBlocks = nonsolidblocks

    def onObservationChanged(self, obs: TimestampedString) -> None:
        self.mc.updateObservations(obs, self.agentId)
        self._observeProcCached()

    def onNewFrameCallback(self, frame: TimestampedVideoFrame) -> None:
        if frame.frametype == FrameType.COLOUR_MAP:
            self.mc.updateSegmentation(frame, self.agentId)
            self._update_cache('getSegmentationFrame')
        else:
            self.mc.updateFrame(frame, self.agentId)
            assert self.mc.getImageFrame() is not None
            self._update_cache('getImageFrame')

    def getVersion(self):
        return self.mc.getVersion()

    def update_in_background(self, time_sleep=0.05):
        self._time_sleep = time_sleep
        self.thread = threading.Thread(target=self.__update_in_background, daemon=True)
        self.thread.start()

    def __update_in_background(self):
        while True:
            self.observeProcCached()
            time.sleep(self._time_sleep)

    def clear(self):
        with self.lock:
            self.cached = {k: (None, 0) for k in self.cached}
            for event in self.events:
                self.cached[event] = [(None, 0)]

    def getCachedObserve(self, method, key=None, readEvent=True):
        with self.lock:
            val = self.cached[method]
        if method in self.events:
            if readEvent:
                self.readEvents[method] = True
                self.cached[method] = [(None, self.cached[method][-1][1])]
        else:
            val = val[0]
        if key is None:
            return val
        else:
            return val[key] if val is not None and key in val else None

    def __peekCache(self, method):
        with self.lock:
            val = self.cached[method]
        if method in self.events:
            return [x[0] for x in val]
        return [val[0]]

    def observeProcCached(self):
        self._observeProcCached()

    def _observeProcCached(self):
        for method in self.methods:
            self._update_cache(method)

    def _update_cache(self, method):
        t_new = time.time()
        v_new = getattr(self.mc, method)(self.agentId)
        outdated = False
        with self.lock:
            if method not in self.events:
                v, t = self.cached[method]
            else:
                v, t = self.cached[method][-1]
            outdated = abs(t_new - t) > self.max_dt
        if v_new is not None or outdated: # or v is None
            with self.lock:
                self.cached_buffer[method] = self.cached[method]
                if method in self.events:
                    if v is None and v_new is None:
                        self.cached[method].pop()
                    self.cached[method].append((v_new, t_new))
                    self.readEvents[method] = False
                else:
                    self.cached[method] = (v_new, t_new)
            self.changed(method)

    def changed(self, name):
        pass

    def remove_mcprefix_rec(self, data):
        if isinstance(data, str):
            return data.split('.')[-1] if 'minecraft' in data else data
        if isinstance(data, dict):
            r = {}
            for k in data.keys():
                r[self.remove_mcprefix_rec(k)] = self.remove_mcprefix_rec(data[k])
            return r
        if isinstance(data, list):
            return [self.remove_mcprefix_rec(l) for l in data]
        return data

    def getItemsAndRecipesLists(self):
        self.sendCommand('recipes')
        self.sendCommand('item_list')
        time.sleep(1)
        item_list = self.waitNotNoneObserve('getItemList', False)
        recipes = self.remove_mcprefix_rec(self.waitNotNoneObserve('getRecipeList', False))
        return item_list, recipes

    def getBlocksDropsList(self):
        self.sendCommand('blockdrops')
        time.sleep(1)
        triples = self.waitNotNoneObserve('getBlocksDropsList', False)
        return triples

    def sendCommandToFindBlock(self, block_name):
        self.sendCommand(f'find_block {block_name}')

    def __getNonSolidBlocks(self):
        self.sendCommand('solid')
        time.sleep(1)
        nonsolid_blocks = self.remove_mcprefix_rec(self.waitNotNoneObserve('getNonSolidBlocks', False))
        return nonsolid_blocks

    def __get_cached_time(self, method):
        if method in self.events:
            tm = self.cached[method][-1][1]
        else:
            tm = self.cached[method][1]
        return tm

    def waitNotNoneObserve(self, method, updateReq=False, observeReq=True):
        # REM: do not use with 'getLineOfSights'

        tm = self.__get_cached_time(method)
        # Do not force observeProcCached if the value was updated less than tick ago
        if time.time() - tm > self.tick and observeReq:
            self.observeProcCached()
            tm = self.__get_cached_time(method)

        # if updated observation is required, observation time should be changed
        # (is not recommended while moving)
        while all(x is None for x in self.__peekCache(method)) or\
              (self.__get_cached_time(method)  == tm and updateReq):
            time.sleep(self.tick)
            self.observeProcCached()
            if self.__get_cached_time(method) - tm > self.max_dt:
                # It's better to return None rather than hanging too long
                break
        return self.getCachedObserve(method)

    def updateAllObservations(self):
        # we don't require observations to be updated in fact, but we try to do an update
        self.observeProcCached()
        start = time.time()
        while not all([self.cached[method][0] is not None or (method in self.canBeNone or method in self.events) for method in self.methods]):
            nones = []
            time.sleep(self.tick)
            self.observeProcCached()
            now = time.time()
            delta = now - start
            if 2 < delta:
                for method in self.methods:
                    if self.cached[method][0] is None:
                        if method not in self.canBeNone:
                            nones.append(method)
                logger.warning("can't update cache for these methods %s", str(nones))

    def addCommandsToBuffer(self, commanList):
        self.commandBuffer.append(commanList)

    def clearCommandBuffer(self, commanList):
        self.commandBuffer.clear()

    def isCommandPoseChanging(self, command):
        if command[0] in RobustObserver.explicitlyPoseChangingCommands or\
            command[0] in RobustObserver.implicitlyPoseChangingCommands:
            return True
        else:
            return False

    def sendCommand(self, command):
        # TODO isinstance check for list err otherwise
        if isinstance(command, str):
            cmd = command.split(' ')
            self.addCommandsToBuffer(cmd)
            self.mc.sendCommand(command, self.agentId)
        else:
            self.addCommandsToBuffer(command)
            self.mc.sendCommand(' '.join(command), self.agentId)

    # ===== specific methods =====

    def dirToAgentPos(self, pos, observeReq=True):
        aPos = self.waitNotNoneObserve('getAgentPos', observeReq=observeReq)
        return self.mc.dirToPos([aPos[0], aPos[1]+1.66, aPos[2]], pos)

    def blockCenterFromPos(self, pos):
        ''' Get block center assuming that `pos` is in its observed face '''
        aPos = self.getCachedObserve('getAgentPos')
        aPos = [aPos[0], aPos[1] + 1.66, aPos[2]]
        dist = dist_vec(aPos, pos)
        # more slightly inside the block before `int_coord`
        return [int_coord(tx + 0.01 * (tx - ax) / dist) + 0.5 for ax, tx in zip(aPos, pos)]

    def blockCenterFromRay(self):
        los = self.getCachedObserve('getLineOfSights')
        if los is None or 'hitType' not in los or los['hitType'] != 'block':
            return None
        return self.blockCenterFromPos([los['x'], los['y'], los['z']])

    def gridIndexToAbsPos(self, index, observeReq=True):
        [x, y, z] = self.mc.gridIndexToPos(index, self.agentId)
        pos = self.waitNotNoneObserve('getAgentPos', observeReq=observeReq)
        # TODO? Round to the center of the block (0.5)?
        return [x + pos[0], y + pos[1], z + pos[2]]

    def getNearGrid3D(self, observeReq=True):
        grid = self.waitNotNoneObserve('getNearGrid', observeReq=observeReq)
        gridBox = self.mc.getGridBox(self.agentId)
        gridSz = [gridBox[i][1]-gridBox[i][0]+1 for i in range(3)]
        return [[grid[(z+y*gridSz[2])*gridSz[0]:(z+1+y*gridSz[2])*gridSz[0]] \
                 for z in range(gridSz[2])] for y in range(gridSz[1])]

    def getYawDeltas(self, observeReq=True):
        pos = self.waitNotNoneObserve('getAgentPos', observeReq=observeReq)
        return MCConnector.yawDelta(degree2rad(pos[4]))

    def gridInYaw(self, observeReq=True):
        '''Vertical slice of the grid in the line-of-sight direction'''
        grid3D = self.getNearGrid3D(observeReq)
        deltas = self.getYawDeltas(observeReq)
        pos = self.getCachedObserve('getAgentPos')
        if grid3D is None or deltas is None:
            return None
        dimX = len(grid3D[0][0])
        dimZ = len(grid3D[0])
        dimY = len(grid3D)
        deltas[0] /= 4
        deltas[2] /= 4
        objs = []
        for y in range(dimY):
            grid2D = grid3D[y]
            line = []
            x = pos[0]
            z = pos[2]
            x0int = int(x)
            z0int = int(z)
            for t in range(dimX + dimZ):
                # TODO? It works, but when the agent is still partly standing
                # on the previous block, it will not show the next block (on which
                # the agent is formally standing, but which can be air, so the agent
                # will fall down if it moves forward within this block)
                if int(x + deltas[0]) != int(x) or int(z + deltas[2]) != int(z):
                    dxGrid = int(x + deltas[0]) - x0int
                    dzGrid = int(z + deltas[2]) - z0int
                    # FixMe? Works for symmetric grids only
                    if abs(dxGrid)*2+1 >= dimX or abs(dzGrid)*2+1 >= dimZ:
                        break
                    line += [grid2D[dzGrid+(dimZ-1)//2][dxGrid+(dimX-1)//2]]
                x += deltas[0]
                z += deltas[2]
            objs += [line]
        return objs

    def analyzeGridInYaw(self, observeReq=True):
        passableBlocks = self.passableBlocks
        deadlyBlocks = RobustObserver.deadlyBlocks
        gridSlice = self.gridInYaw(observeReq)
        underground = gridSlice[(len(gridSlice) - 1) // 2 - 2]
        ground = gridSlice[(len(gridSlice) - 1) // 2 - 1]
        solid = all([b not in passableBlocks for b in ground])
        wayLv0 = gridSlice[(len(gridSlice) - 1) // 2]
        wayLv1 = gridSlice[(len(gridSlice) - 1) // 2 + 1]
        passWay = all([b in passableBlocks for b in wayLv0]) and \
                  all([b in passableBlocks for b in wayLv1])
        lvl = (len(gridSlice) + 1) // 2
        for h in range(len(gridSlice)):
            if gridSlice[-h-1][0] not in passableBlocks:
                break
            lvl -= 1
        safe = all([b not in deadlyBlocks for b in ground]) and \
               all([b not in deadlyBlocks for b in wayLv0]) and \
               all([b not in deadlyBlocks for b in wayLv1])
        if lvl < -1:
            safe = safe and all([b not in deadlyBlocks for b in underground])
            if ground[0] != 'water' and underground[0] != 'water':
                safe = False
        return {'solid': solid, 'passWay': passWay, 'level': lvl, 'safe': safe}

    def craft(self, item):
        self.sendCommand('craft ' + item)
        # always sleep after crafting, so information about the crafted item
        # will be received
        time.sleep(0.2)

    def stopMove(self):
        self.mc.move("0", self.agentId)
        self.mc.turn("0", self.agentId)
        self.mc.pitch("0", self.agentId)
        self.mc.jump("0", self.agentId)
        self.mc.strafe("0", self.agentId)

    def filterInventoryItem(self, item, observeReq=True):
        inv = self.waitNotNoneObserve('getInventory', True, observeReq=observeReq)
        return list(filter(lambda entry: entry['type']==item, inv))

    def softFilterInventoryItem(self, item, observeReq=True):
        inv = self.waitNotNoneObserve('getInventory', True, observeReq=observeReq)
        return list(filter(lambda entry: item in entry['type'], inv))

    def nearestFromGrid(self, objs, observeReq=True, return_target_block=False):
        if not isinstance(objs, list):
            objs = [objs]
        grid = self.waitNotNoneObserve('getNearGrid', observeReq=observeReq)
        pos  = self.waitNotNoneObserve('getAgentPos', observeReq=observeReq)
        d2 = 10000
        target = None
        for i in range(len(grid)):
            if grid[i] not in objs: continue
            [x, y, z] = self.mc.gridIndexToPos(i)
            # penalty for height
            d2c = x * x + (y - 1.66) * (y - 1.66) * 4 + z * z
            if d2c < d2:
                d2 = d2c
                # target = self.gridIndexToAbsPos(i, observeReq)
                target = [x + pos[0], y + pos[1], z + pos[2]]
                if return_target_block:
                    target = [target, grid[i]]
        return target

    def nearestFromEntities(self, obj, observeReq=True):
        ent = self.waitNotNoneObserve('getNearEntities', observeReq=observeReq)
        pos = self.waitNotNoneObserve('getAgentPos', observeReq=observeReq)
        d2 = 10000
        target = None
        for e in ent:
            if e['name'] != obj: continue
            [x, y, z] = [e['x'], e['y'], e['z']]
            if abs(y - pos[1]) > 1: continue
            d2c = (x - pos[0]) * (x - pos[0]) + (y - pos[1]) * (y - pos[1]) + (z - pos[2]) * (z - pos[2])
            if d2c < d2:
                d2 = d2c
                target = [x, y, z]
        return target


class RobustObserverWithCallbacks(RobustObserver):
    def __init__(self, mc, agentId=0):
        super().__init__(mc, agentId)
        # name, on_change, function triples
        self.callbacks = []
        # future -> name, cb pairs
        self._futures = dict()
        self._in_process = set()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.mlogy = None

    def set_mlogy(self, mlogy):
        self.mlogy = mlogy

    def changed(self, name):
        for (cb_name, on_change, cb) in self.callbacks:
            if name == on_change and cb not in self._in_process:
                # submit to the thread pool
                self.submit(cb, cb_name)

    def addCallback(self, name, on_change, cb):
        """
        add callback to be called if data in robust observer's cache
        is changed

        name: str
           name of callback, it is used to store returned data in the cache
           if None returned value won't be saved
        on_change: str
           key to be monitored for change event
        cb: Callable
           callback
        """
        if name is not None:
            self.cached[name] = (None, 0)
        self.callbacks.append((name, on_change, cb))

    def done_callback(self, fut):
        if fut in self._futures:
            tm = time.time()
            with self.lock:
                name, cb = self._futures[fut]
                del self._futures[fut]
                result = None
                exception = fut.exception()
                if exception is None:
                    result = fut.result()
                else:
                    logger.exception(exception)
                if name is not None:
                    # logger.debug('adding results from %s', name)
                    self.cached[name] = (result, tm)
                    self.changed(name)
                self._in_process.discard(cb)

    def submit(self, cb, name):
        future = self.executor.submit(cb)
        with self.lock:
            self._futures[future] = (name, cb)
            self._in_process.add(cb)
        future.add_done_callback(self.done_callback)

