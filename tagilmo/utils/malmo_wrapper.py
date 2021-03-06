import json
import uuid
import time
import math
import sys

import MalmoPython

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
                self.mission_record.recordMP4(24,2000000)
        self.client_pool = MalmoPython.ClientPool()
        for x in range(10000, 10000 + nAgents):
            self.client_pool.add( MalmoPython.ClientInfo(serverIp, x) )
        self.worldStates = [None]*nAgents
        self.observe = [None]*nAgents
        self.isAlive = [True] * nAgents

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

    def observeProc(self, nAgent=None):
        r = range(len(self.agent_hosts)) if nAgent is None else range(nAgent, nAgent+1)
        for n in r:
            self.worldStates[n] = self.agent_hosts[n].getWorldState()
            self.isAlive[n] = self.worldStates[n].is_mission_running
            obs = self.worldStates[n].observations
            self.observe[n] = json.loads(obs[-1].text) if len(obs) > 0 else None

    def getAgentPos(self, nAgent=0):
        if (self.observe[nAgent] is not None) and ('XPos' in self.observe[nAgent]):
            return [self.observe[nAgent][key] for key in ['XPos', 'YPos', 'ZPos', 'Pitch', 'Yaw']]
        else:
            return None

    def getFullStat(self, key, nAgent=0):
        # keys (position): 'XPos', 'YPos', 'ZPos', 'Pitch', 'Yaw'
        # keys (status)  : 'Life', 'Food', 'Air', 'IsAlive'
        # keys (more)    : 'XP', 'Score', 'Name', 'WorldTime', 'TotalTime', 'DistanceTravelled', 'TimeAlive', 'MobsKilled', 'PlayersKilled', 'DamageTaken', 'DamageDealt'
        if (self.observe[nAgent] is not None) and (key in self.observe[nAgent]):
            return self.observe[nAgent][key]
        else:
            return None

    def isLineOfSightAvailable(self, nAgent=0):
        return not self.observe[nAgent] is None and 'LineOfSight' in self.observe[nAgent]

    def isInventoryAvailable(self, nAgent=0):
        return not self.observe[nAgent] is None and 'inventory' in self.observe[nAgent]

    def getLineOfSight(self, key, nAgent=0):
        # keys: 'hitType', 'x', 'y', 'z', 'type', 'prop_snowy', 'inRange', 'distance'
        if self.isLineOfSightAvailable(nAgent) and (key in self.observe[nAgent]['LineOfSight']):
            return self.observe[nAgent]['LineOfSight'][key]
        else:
            return None
    
    def sendCommand(self, command, nAgent=0):
        self.agent_hosts[nAgent].sendCommand(command)

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

    def getGridBox(self, nAgent=0):
        return self.missionDesc.agentSections[nAgent].agenthandlers.observations.gridNear

    def getNearGrid3D(self, nAgent=0):
        if (self.observe[nAgent] is not None) and ('grid_near' in self.observe[nAgent]):
            grid = self.observe[nAgent]['grid_near']
            gridBox = self.getGridBox(nAgent)
            gridSz = [gridBox[i][1]-gridBox[i][0]+1 for i in range(3)]
            return [[grid[(z+y*gridSz[2])*gridSz[0]:(z+1+y*gridSz[2])*gridSz[0]] for z in range(gridSz[2])] for y in range(gridSz[1])]
        else:
            return None

    def getYawDeltas(self, nAgent=0):
        pos = self.getAgentPos(nAgent)
        if pos is not None:
            return MalmoConnector.yawDelta(pos[4] * math.pi / 180.)
        else:
            return None
    
    def dirToPos(self, pos, nAgent=0):
        aPos = self.getAgentPos(nAgent)
        if aPos is None: return None
        dx = pos[0] - aPos[0]
        dz = pos[2] - aPos[2]
        yaw = -math.atan2(dx, dz)
        pitch = -math.atan2(pos[1] - aPos[1] - 1, math.sqrt(dx * dx + dz * dz))
        return [pitch, yaw]

    def gridIndexToPos(self, index, nAgent=0):
        gridBox = self.getGridBox(nAgent)
        gridSz = [gridBox[i][1]-gridBox[i][0]+1 for i in range(3)]
        y = index // (gridSz[0] * gridSz[2])
        index -= y * (gridSz[0] * gridSz[2])
        y += gridBox[1][0]
        z = index // gridSz[0] + gridBox[2][0]
        x = index % gridSz[0] + gridBox[0][0]
        return [x, y, z]


    def gridIndexToAbsPos(self, index, nAgent=0):
        [x, y, z] = self.gridIndexToPos(index, nAgent)
        pos = self.getAgentPos(nAgent)
        if pos is None: return None
        # TODO? Round to the center of the block (0.5)?
        return [x + pos[0], y + pos[1], z + pos[2]]

    def gridInYaw(self, nAgent=0):
        '''Vertical slice of the grid in the line-of-sight direction'''
        grid3D = self.getNearGrid3D(nAgent)
        deltas = self.getYawDeltas(nAgent)
        pos = self.getAgentPos(nAgent)
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

    def filterInventoryItem(self, item, nAgent=0):
        if not self.isInventoryAvailable(nAgent):
            return None
        return list(filter(lambda entry: entry['type']==item, self.observe[nAgent]['inventory']))
