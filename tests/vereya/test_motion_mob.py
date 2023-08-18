import logging
import json
import time
import unittest
import tagilmo.utils.mission_builder as mb
from tagilmo.utils.vereya_wrapper import MCConnector, RobustObserver
from base_test import BaseTest
from tagilmo import VereyaPython
from common import init_mission
from test_motion_vereya import TestMotion


logger = logging.getLogger(__file__)

class TestMotionMob(TestMotion):

    mc = None
    obs = None

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        start = (-151.0, -213.0)
        mc, obs = init_mission(None, start_x=start[0], start_z=start[1], forceReset='true', seed='43')
        cls.mc = mc
        cls.obs = obs
        assert mc.safeStart()
        time.sleep(4)
        # create mob above player
        x, y, z = obs.waitNotNoneObserve('getAgentPos')[:3]
        mc.sendCommand(f"chat /summon minecraft:chicken {x} {y + 7} {z} {{NoAI:1}}")
        time.sleep(3)
        key = None
        for key in mc.getControlledMobs():
            mc.agentId = key
            obs.agentId = key
        assert key is not None


del TestMotion

if __name__ == '__main__':
    VereyaPython.setupLogger()
    unittest.main()
