import logging
import json
import time
import unittest
import tagilmo.utils.mission_builder as mb
from tagilmo.utils.vereya_wrapper import MCConnector, RobustObserver
from base_test import BaseTest
from tagilmo import VereyaPython
from common import init_mission


logger = logging.getLogger(__file__)

class TestMotion(BaseTest):

    mc = None
    obs = None

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        start = (-151.0, -213.0)
        mc, obs = init_mission(None, start_x=start[0], start_y=start[1], forceReset='true', seed='43')
        cls.mc = mc
        cls.obs = obs
        assert mc.safeStart()
        time.sleep(4)

    def test_motion_forward(self):
        obs = self.obs
        mc = self.mc
        pos0 = obs.getCachedObserve('getAgentPos')
        logger.info('position ' + str(pos0))
        print('sending command move 1')
        mc.move(1)
        time.sleep(1)

        pos_move1 = obs.getCachedObserve('getAgentPos')
        logger.info('position ' + str(pos_move1))
        self.assertNotEqual(pos0, pos_move1, 'move 1 sent but position is the same')
        print('sending command move 0')
        mc.move(0)
        time.sleep(0.5)

        pos_move0 = obs.getCachedObserve('getAgentPos')
        time.sleep(1)
        pos1_move0 = obs.getCachedObserve('getAgentPos')
        logger.info('position ' + str(pos1_move0))
        time.sleep(1)
        pos2_move0 = obs.getCachedObserve('getAgentPos')
        logger.info('position ' + str(pos2_move0))
        self.assertEqual(pos1_move0, pos2_move0)

    def test_rotate(self):
        obs = self.obs
        mc = self.mc

        PY = slice(3, None)
        XYZ = slice(None, 3)
        print('sending command turn 0.1')
        pos0 = obs.getCachedObserve('getAgentPos')
        mc.turn(0.1)
        time.sleep(1)
        pos1 = obs.getCachedObserve('getAgentPos')

        print('sending command turn 0')
        mc.turn(0)
        time.sleep(1)
        pos2 = obs.getCachedObserve('getAgentPos')
        self.assertNotEqual(pos2[PY], pos1[PY])
        self.assertEqual(pos2[XYZ], pos0[XYZ])


    def teadDown(self):
        self.obs.stopMove()
        super().tearDown()

    def setUp(self):
        self.obs.stopMove()
        super().setUp()

    @classmethod
    def tearDownClass(cls, *args, **kwargs):
        cls.mc.stop()


if __name__ == '__main__':
    VereyaPython.setupLogger()
    unittest.main()
