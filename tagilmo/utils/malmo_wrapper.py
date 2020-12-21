import json
import uuid
import time
import sys

import MalmoPython

import tagilmo.utils.malmoutils as malmoutils
from tagilmo.utils.mission_builder import MissionXML


class MalmoConnector:

    def __init__(self, missionXML, serverIp='127.0.0.1'):
        self.mission = MalmoPython.MissionSpec(missionXML.xml(), True)
        self.mission_record = MalmoPython.MissionRecordSpec()
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

    def getLineOfSight(self, key, nAgent=0):
        # keys: 'hitType', 'x', 'y', 'z', 'type', 'prop_snowy', 'inRange', 'distance'
        if (self.observe[nAgent] is not None) and (u'LineOfSight' in self.observe[nAgent]) and (key in self.observe[nAgent][u'LineOfSight']):
            return self.observe[nAgent][u'LineOfSight'][key]
        else:
            return None
    
    def sendCommand(self, command, nAgent=0):
        self.agent_hosts[nAgent].sendCommand(command)

    #TODO? Extend API?
    def getNearEntities(self, nAgent=0):
        if (self.observe[nAgent] is not None) and (u'ents_near' in self.observe[nAgent]):
            return self.observe[nAgent][u'ents_near']
        else:
            return None

    #TODO? Extend API (e.g. convert to 3D array)?
    def getNearGrid(self, nAgent=0):
        if (self.observe[nAgent] is not None) and (u'grid_near' in self.observe[nAgent]):
            return self.observe[nAgent][u'grid_near']
        else:
            return None

        
