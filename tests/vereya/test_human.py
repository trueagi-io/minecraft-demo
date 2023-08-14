import unittest
import logging
from tagilmo import VereyaPython
import json
import time
import tagilmo.utils.mission_builder as mb
from tagilmo.utils.vereya_wrapper import MCConnector, RobustObserver
from common import init_mission

class TestQuit(unittest.TestCase):
    mc: MCConnector = None

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        start = (-108.0, 78, -187.0)
        mc, obs = init_mission(None, start_x=start[0], start_y=start[1], start_z=start[2], seed='5', forceReset="true")
        cls.mc = mc
        assert mc.safeStart()
        time.sleep(3)

    @classmethod
    def tearDownClass(cls, *args, **kwargs):
        cls.mc.stop()

    def setUp(self):
        self.mc.sendCommand("chat /clear")
        time.sleep(3)

    def test_human(self):
        self.assertTrue(self.mc.is_mission_running())
        print('send ochat')
        while True:
            time.sleep(0.02)
            self.mc.observeProc()
            if self.mc.observe[0] is None:
                continue
            obs = self.mc.getHumanInputs()
            if obs is not None:
                print(obs)
        self.mc.sendCommand("quit")
        time.sleep(4)
        self.assertFalse(self.mc.is_mission_running())


def main():
    VereyaPython.setupLogger()
    unittest.main()
        
if __name__ == '__main__':
   main()
