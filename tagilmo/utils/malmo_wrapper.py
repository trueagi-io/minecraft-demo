import json
import uuid
import time
import math
import sys

import MalmoPython

import numpy
import tagilmo.utils.malmoutils as malmoutils
from tagilmo.utils.mission_builder import MissionXML



class MalmoConnector:

    @staticmethod
    def yawDelta(yawRad):
        return [-math.sin(yawRad), 0, math.cos(yawRad)]

    def setMissionXML(self, missionXML):
        self.missionDesc = missionXML
        self.mission = MalmoPython.MissionSpec(missionXML.xml(), True)
        self.mission_record = MalmoPython.MissionRecordSpec()

    def __init__(self, missionXML, serverIp='127.0.0.1'):
        self.missionDesc = None
        self.mission = None
        self.mission_record = None
        self.setMissionXML(missionXML)
        nAgents = len(missionXML.agentSections)
        self.agent_hosts = []
        self.agent_hosts += [MalmoPython.AgentHost() for n in range(nAgents)]
        self.agent_hosts[0].parse( sys.argv )
        if self.receivedArgument('recording_dir'):
            recordingsDirectory = malmoutils.get_recordings_directory(self.agent_hosts[0])
            self.mission_record.recordRewards()
            self.mission_record.recordObservations()
            self.mission_record.recordCommands()
            self.mission_record.setDestination(recordingsDirectory + "//" + "lastRecording.tgz")
            if self.agent_hosts[0].receivedArgument("record_video"):
                self.mission_record.recordMP4(24, 2000000)
        self.client_pool = MalmoPython.ClientPool()
        for x in range(10000, 10000 + nAgents+5):
            self.client_pool.add( MalmoPython.ClientInfo(serverIp, x) )
        self.worldStates = [None]*nAgents
        self.observe = [None]*nAgents
        self.isAlive = [True] * nAgents
        self.pixels = [None] * nAgents 
        self.segmentation = [None] * nAgents

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
                except MalmoPython.MissionException as e:
                    errorCode = e.details.errorCode
                    if errorCode == MalmoPython.MissionErrorCode.MISSION_SERVER_WARMING_UP:
                        print("Server not quite ready yet - waiting...")
                        time.sleep(2)
                    elif errorCode == MalmoPython.MissionErrorCode.MISSION_INSUFFICIENT_CLIENTS_AVAILABLE:
                        print("Not enough available Minecraft instances running.")
                        used_attempts += 1
                        if used_attempts < max_attempts:
                            print("Will wait in case they are starting up.", max_attempts - used_attempts, "attempts left.")
                            time.sleep(2)
                    elif errorCode == MalmoPython.MissionErrorCode.MISSION_SERVER_NOT_FOUND:
                        print("Server not found - has the mission with role 0 been started yet?")
                        used_attempts += 1
                        if used_attempts < max_attempts:
                            print("Will wait and retry.", max_attempts - used_attempts, "attempts left.")
                            time.sleep(2)
                    elif errorCode == MalmoPython.MissionErrorCode.MISSION_ALREADY_RUNNING:
                        print('The mission already running')
                        return True
                    else:
                        print("Other error:", e.message)
                        return False
                if used_attempts == max_attempts:
                    print("All attempts to start the mission are failed.")
                    return False
        # waiting for the real start
        start_flags = [False for a in self.agent_hosts]
        start_time = time.time()
        time_out = 120  # Allow a two minute timeout.
        while not all(start_flags) and time.time() - start_time < time_out:
            states = [a.peekWorldState() for a in self.agent_hosts]
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

    def is_mission_running(self, nAgent=0):
        world_state = self.agent_hosts[nAgent].getWorldState()
        return world_state.is_mission_running

    def sendCommand(self, command, nAgent=0):
        self.agent_hosts[nAgent].sendCommand(command)

    def observeProc(self, nAgent=None):
        r = range(len(self.agent_hosts)) if nAgent is None else range(nAgent, nAgent+1)
        for n in r:
            self.worldStates[n] = self.agent_hosts[n].getWorldState()
            self.isAlive[n] = self.worldStates[n].is_mission_running
            obs = self.worldStates[n].observations
            self.observe[n] = json.loads(obs[-1].text) if len(obs) > 0 else None
            # might need to wait for a new frame
            frames = self.worldStates[n].video_frames
            segments = self.worldStates[n].video_frames_colourmap
            if frames:
                self.pixels[n] = numpy.frombuffer(frames[0].pixels, dtype=numpy.uint8)
            else:
                self.pixels[n] = None
            if segments:
                self.segmentation[n] = numpy.frombuffer(segments[0].pixels, dtype=numpy.uint8)
            else:
                self.segmentation[n] = None

    def getImage(self, nAgent=0):
        return self.pixels[nAgent]

    def getSegmentation(self, nAgent=0):
        if self.segmentation:
            return self.segmentation[nAgent]

    def getAgentPos(self, nAgent=0):
        if (self.observe[nAgent] is not None) and ('XPos' in self.observe[nAgent]):
            return [self.observe[nAgent][key] for key in ['XPos', 'YPos', 'ZPos', 'Pitch', 'Yaw']]
        else:
            return None

    def getFullStat(self, key, nAgent=0):
        """
            Dsc: this function was intended to return observations from full stats.
                 However, full stats don't have a dedicated key, but are placed directly
                 (each stat has its own key). Thus, this procedure can return any observation.
            FullStat:
                keys (position): 'XPos', 'YPos', 'ZPos', 'Pitch', 'Yaw'
                keys (status)  : 'Life', 'Food', 'Air', 'IsAlive'
                keys (more)    : 'XP', 'Score', 'Name', 'WorldTime', 'TotalTime', 'DistanceTravelled', 'TimeAlive',
                                 'MobsKilled', 'PlayersKilled', 'DamageTaken', 'DamageDealt'
        """
        if (self.observe[nAgent] is not None) and (key in self.observe[nAgent]):
            return self.observe[nAgent][key]
        else:
            return None

    def isLineOfSightAvailable(self, nAgent=0):
        return not self.observe[nAgent] is None and 'LineOfSight' in self.observe[nAgent]

    def getLineOfSights(self, nAgent=0):
        return self.observe[nAgent]['LineOfSight'] if self.isLineOfSightAvailable(nAgent) else None

    def getLineOfSight(self, key, nAgent=0):
        # keys: 'hitType', 'x', 'y', 'z', 'type', 'prop_snowy', 'inRange', 'distance'
        los = self.getLineOfSights(nAgent)
        return los[key] if los is not None and key in los else None
    
    def getNearEntities(self, nAgent=0):
        if (self.observe[nAgent] is not None) and ('ents_near' in self.observe[nAgent]):
            return self.observe[nAgent]['ents_near']
        else:
            return None

    def getNearGrid(self, nAgent=0):
        if (self.observe[nAgent] is not None) and ('grid_near' in self.observe[nAgent]):
            return self.observe[nAgent]['grid_near']
        else:
            return None

    def getLife(self, nAgent=0):
        if (self.observe[nAgent] is not None) and ('Life' in self.observe[nAgent]):
            return self.observe[nAgent]['Life']
        else:
            return None

    def isInventoryAvailable(self, nAgent=0):
        return not self.observe[nAgent] is None and 'inventory' in self.observe[nAgent]

    def getInventory(self, nAgent=0):
        return self.observe[nAgent]['inventory'] if self.isInventoryAvailable(nAgent) else None

    def getGridBox(self, nAgent=0):
        return self.missionDesc.agentSections[nAgent].agenthandlers.observations.gridNear

    def gridIndexToPos(self, index, nAgent=0):
        gridBox = self.getGridBox(nAgent)
        gridSz = [gridBox[i][1]-gridBox[i][0]+1 for i in range(3)]
        y = index // (gridSz[0] * gridSz[2])
        index -= y * (gridSz[0] * gridSz[2])
        y += gridBox[1][0]
        z = index // gridSz[0] + gridBox[2][0]
        x = index % gridSz[0] + gridBox[0][0]
        return [x, y, z]

    def dirToPos(self, aPos, pos):
        dx = pos[0] - aPos[0]
        dz = pos[2] - aPos[2]
        yaw = -math.atan2(dx, dz)
        pitch = -math.atan2(pos[1] - aPos[1] - 1, math.sqrt(dx * dx + dz * dz))
        return [pitch, yaw]


class RobustObserver:

    passableBlocks = ['air', 'water', 'lava', 'double_plant', 'tallgrass', 'reeds', 'red_flower', 'yellow_flower']
    deadlyBlocks = ['lava']

    def __init__(self, mc, nAgent = 0):
        self.mc = mc
        self.nAgent = nAgent
        self.tick = 0.02
        self.max_dt = 1.0
        self.methods = ['getNearEntities', 'getNearGrid', 'getAgentPos', 'getLineOfSights',
                        'getLife', 'getInventory']
        self.canBeNone = ['getLineOfSights']
        self.cached = {method : (None, 0) for method in self.methods}
    
    def getCachedObserve(self, method, key = None):
        val = self.cached[method][0]
        if key is None:
            return val
        else:
            return val[key] if val is not None and key in val else None

    def observeProcCached(self):
        self.mc.observeProc()
        t_new = time.time()
        for method in self.methods:
            v_new = getattr(self.mc, method)(self.nAgent)
            v, t = self.cached[method]
            if v_new is not None or t_new - t > self.max_dt: # or v is None
                self.cached[method] = (v_new, t_new)

    def waitNotNoneObserve(self, method, updateReq=False, observeReq=True):
        # REM: do not use with 'getLineOfSights'
        tm = self.cached[method][1]
        # Do not force observeProcCached if the value was updated less than tick ago
        if time.time() - tm > self.tick and observeReq:
            self.observeProcCached()
        # if updated observation is required, observation time should be changed
        # (is not recommended while moving)
        while self.getCachedObserve(method) is None or\
              (self.cached[method][1] == tm and updateReq):
            time.sleep(self.tick)
            self.observeProcCached()
            if self.cached[method][1] - tm > self.max_dt:
                # It's better to return None rather than hanging too long
                break
        return self.getCachedObserve(method)
    
    def updateAllObservations(self):
        # we don't require observations to be updated in fact, but we try to do an update
        self.observeProcCached()
        while not all([self.cached[method][0] is not None or method in self.canBeNone for method in self.methods]):
            time.sleep(self.tick)
            self.observeProcCached()
    
    def sendCommand(self, command):
        self.mc.sendCommand(command, self.nAgent)

    # ===== specific methods =====

    def dirToAgentPos(self, pos, observeReq=True):
        aPos = self.waitNotNoneObserve('getAgentPos', observeReq=observeReq)
        return self.mc.dirToPos(aPos, pos)

    def gridIndexToAbsPos(self, index, observeReq=True):
        [x, y, z] = self.mc.gridIndexToPos(index, self.nAgent)
        pos = self.waitNotNoneObserve('getAgentPos', observeReq=observeReq)
        # TODO? Round to the center of the block (0.5)?
        return [x + pos[0], y + pos[1], z + pos[2]]

    def getNearGrid3D(self, observeReq=True):
        grid = self.waitNotNoneObserve('getNearGrid', observeReq=observeReq)
        gridBox = self.mc.getGridBox(self.nAgent)
        gridSz = [gridBox[i][1]-gridBox[i][0]+1 for i in range(3)]
        return [[grid[(z+y*gridSz[2])*gridSz[0]:(z+1+y*gridSz[2])*gridSz[0]] \
                 for z in range(gridSz[2])] for y in range(gridSz[1])]

    def getYawDeltas(self, observeReq=True):
        pos = self.waitNotNoneObserve('getAgentPos', observeReq=observeReq)
        return MalmoConnector.yawDelta(pos[4] * math.pi / 180.)
    
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
        passableBlocks = RobustObserver.passableBlocks
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
        self.sendCommand("move 0")
        self.sendCommand("turn 0")
        self.sendCommand("pitch 0")
        self.sendCommand("jump 0")
        self.sendCommand("strafe 0")
        # self.sendCommand("attack 0")

    def filterInventoryItem(self, item, observeReq=True):
        inv = self.waitNotNoneObserve('getInventory', True, observeReq=observeReq)
        return list(filter(lambda entry: entry['type']==item, inv))

    def nearestFromGrid(self, obj, observeReq=True):
        grid = self.waitNotNoneObserve('getNearGrid', observeReq=observeReq)
        pos  = self.waitNotNoneObserve('getAgentPos', observeReq=observeReq)
        d2 = 10000
        target = None
        for i in range(len(grid)):
            if grid[i] != obj: continue
            [x, y, z] = self.mc.gridIndexToPos(i)
            d2c = x * x + y * y + z * z
            if d2c < d2:
                d2 = d2c
                # target = self.gridIndexToAbsPos(i, observeReq)
                target = [x + pos[0], y + pos[1], z + pos[2]]
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



