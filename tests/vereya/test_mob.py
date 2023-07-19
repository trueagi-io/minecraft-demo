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
import random


logger = logging.getLogger(__name__)


class TestData(BaseTest):
    mc = None
    rob = None

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        start = (-126, 7.0)
        mc, obs = init_mission(None, start_x=start[0], start_y=start[1], seed='43', forceReset=True)

        cls.mc = mc
        cls.rob = obs
        cls.rob_c = RobustObserver(mc)
        mc.safeStart()
        time.sleep(1)

    @classmethod
    def tearDownClass(cls, *args, **kwargs):
        cls.mc.stop()

    def setUp(self):
        super().setUp()
        time.sleep(2)

    def test_move_player_chicken(self):
        # create mob about player
        x, y, z = self.rob.waitNotNoneObserve('getAgentPos')[:3] 
        self.mc.sendCommand(f"chat /summon minecraft:chicken {x} {y + 3} {z + 3} {{NoAI:1}}")
        # sleep while chicken is falling
        time.sleep(4)
        controlled = self.rob.getCachedObserve("getControlledMobs")
        id = list(controlled.keys())[0]
        self.rob_c.agentId = id
        self.rob_c.clear()
        time.sleep(1)
        pos_chicken0 = self.rob_c.getCachedObserve('getAgentPos')
        pos_player0 = self.rob.getCachedObserve('getAgentPos')
        self.mc.move("-0.3", id)
        self.mc.move("-0.3")
        time.sleep(1)
        self.rob.stopMove()
        self.rob_c.stopMove()
        time.sleep(0.5)
        pos_chicken1 = self.rob_c.getCachedObserve('getAgentPos')
        pos_player1 = self.rob.getCachedObserve('getAgentPos')
        diff_player = abs(pos_player1[2] - pos_player0[2])
        diff_chicken = abs(pos_chicken1[2] - pos_chicken0[2])
        print(diff_player, diff_chicken)
        self.assertTrue(0.1 < diff_player, f"test player moved {diff_player} blocks")
        self.assertTrue(0.1 < diff_chicken, f"test chicken moved {diff_chicken} blocks")
 

def main():
    VereyaPython.setupLogger()
    unittest.main()


if __name__ == '__main__':
   main()

