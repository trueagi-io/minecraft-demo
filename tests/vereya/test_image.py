import os
import unittest
import logging
from tagilmo import VereyaPython
import json
import time
import tagilmo.utils.mission_builder as mb
from tagilmo.utils.vereya_wrapper import MCConnector, RobustObserver
from base_test import BaseTest
from common import init_mission


logger = logging.getLogger(__name__)


class TestData(BaseTest):
    mc = None
    rob = None

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        start = (-108.0, -187.0)
        mc, obs = init_mission(None, start_x=start[0], start_z=start[1], seed='5', forceReset='true')
        cls.mc = mc
        cls.rob = obs
        assert mc.safeStart()
        time.sleep(4)

    @classmethod
    def tearDownClass(cls, *args, **kwargs):
        cls.mc.stop()

    def setUp(self):
        super().setUp()
        time.sleep(2)

    def test_get_image(self):
        img_frame = self.rob.waitNotNoneObserve('getImageFrame')
        self.assertTrue(img_frame is not None)
        m = img_frame.pixels.mean()
        self.mc.sendCommand('turn 0.1')
        time.sleep(2)
        img_frame1 = self.rob.waitNotNoneObserve('getImageFrame')
        self.assertTrue(img_frame1 is not None)
        m1 = img_frame1.pixels.mean()
        self.assertNotEqual(m, m1, "mean should be equal")


        
def main():
    VereyaPython.setupLogger()
    unittest.main()

        
if __name__ == '__main__':
   main()
