import unittest
import logging
from tagilmo import VereyaPython
import json
import time
import tagilmo.utils.mission_builder as mb
from tagilmo.utils.vereya_wrapper import MCConnector, RobustObserver
from base_test import BaseTest


def init_mission(mc, start_x=None, start_z=None):
    want_depth = False

    video_producer = mb.VideoProducer(width=320 * 4,
                                      height=240 * 4, want_depth=want_depth)

    obs = mb.Observations()

    agent_handlers = mb.AgentHandlers(observations=obs,
            video_producer=video_producer)

    print('starting at ({0}, {1})'.format(start_x, start_z))

    #miss = mb.MissionXML(namespace="ProjectMalmo.microsoft.com",
    miss = mb.MissionXML(
                         agentSections=[mb.AgentSection(name='Cristina',
             agenthandlers=agent_handlers,
                                      #    depth
             agentstart=mb.AgentStart([start_x, -60.0, start_z, 1]))])
    world = mb.flatworld("",
        seed='5',
        forceReset="false")
    miss.setWorld(world)
    miss.serverSection.initial_conditions.allowedmobs = "Pig Sheep Cow Chicken Ozelot Rabbit Villager"
    # uncomment to disable passage of time:
    miss.serverSection.initial_conditions.time_pass = 'false'
    miss.serverSection.initial_conditions.time_start = "1000"

    if mc is None:
        mc = MCConnector(miss)
        obs = RobustObserver(mc)
    else:
        mc.setMissionXML(miss)
    return mc, obs


class TestActionStatus(BaseTest):
    mc = None

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        start = (-108.0, -187.0)
        mc, obs = init_mission(None, start_x=start[0], start_z=start[1])
        cls.mc = mc
        assert mc.safeStart()
        time.sleep(3)

    @classmethod
    def tearDownClass(cls, *args, **kwargs):
        cls.mc.stop()

    def setUp(self):
        super().setUp()
        self.mc.sendCommand("chat /clear")
        time.sleep(3)

    def test_action_status(self):
        targetActionStatus = {"move" : 0.4, "jump" : 1, "turn" : -0.2,
                              "crouch" : 1, "use" : 0, "attack" : 1, 
                              "pitch" : 0.3, "strafe" : 1}
        for action in targetActionStatus:
            self.mc.sendCommand(f"{action} {targetActionStatus[action]}")
            time.sleep(0.5)
        self.mc.observeProc()
        actionStatus = self.mc.getActionStatus()
        self.assertEqual(targetActionStatus, actionStatus)
        for action in targetActionStatus:
            targetActionStatus[action] = 0
        for action in targetActionStatus:
            self.mc.sendCommand(f"{action} {targetActionStatus[action]}")
            time.sleep(0.5)
        self.mc.observeProc()
        actionStatus = self.mc.getActionStatus()
        self.assertEqual(targetActionStatus, actionStatus)

def main():
    VereyaPython.setupLogger()
    unittest.main()
        
if __name__ == '__main__':
   main()
