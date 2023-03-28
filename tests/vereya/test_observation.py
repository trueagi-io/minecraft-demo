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
        start = (-125.0, 73.0)
        mc, obs = init_mission(None, start_x=start[0], start_y=start[1], seed='4')
        cls.mc = mc
        cls.rob = obs
        mc.safeStart()
        time.sleep(1)

    @classmethod
    def tearDownClass(cls, *args, **kwargs):
        cls.mc.stop()

    def setUp(self):
        super().setUp()
        time.sleep(2)

    def test_observation_from_ray(self):
        dist = self.getDist()
        self.assertGreater(dist, 0)

    def test_observation_from_chat(self):
        logger.info("send chat")
        self.mc.sendCommand("chat get wooden_axe")
        logger.info("wait chat")
        start = time.time()
        while True:
            command = self.rob.waitNotNoneObserve('getChat', observeReq=False)
            if command is not None:
                break
            time.sleep(0.05)
            end = time.time()
            if end - start > 2:
                break
        logger.info("result chat " + str(command))
        self.assertEqual(command[0], "get wooden_axe")

    def test_observation_from_item(self):
        item_list, recipes = self.rob.getItemsAndRecipesLists()
        self.assertGreater(len(item_list), 0, "item_list len")
        self.assertGreater(len(recipes), 0, "recipes len")

    def test_observation_from_triples(self):
        blockdrops = self.rob.getBlocksDropsList()
        self.assertGreater(len(blockdrops), 0, "blockdrops len")
        for blockdrop in blockdrops:
            if blockdrop['block_name'] == 'diamond_ore' and blockdrop['item_name'] == 'diamond_ore':
                self.assertEqual(blockdrop['tool'], 'silkt_iron_pickaxe', "check diamond_ore tool")
            if blockdrop['block_name'] == 'iron_ore' and blockdrop['tool'] == 'stone_pickaxe':
                self.assertEqual(blockdrop['item_name'], 'raw_iron', "check iron")
            if blockdrop['block_name'] == 'birch_leaves' and blockdrop['tool'] == 'shears':
                self.assertEqual(blockdrop['item_name'], 'birch_leaves', "check leaves")

    def test_game_state(self):
        self.mc.observeProc()
        self.assertTrue(self.mc.getFullStat(key="isPaused") is not None)
        self.assertTrue(self.mc.getFullStat(key="input_type") is not None)

    def getDist(self):
        mc = self.mc
        c = 0
        prev_pitch = None
        while True:
            mc.observeProc()
            pos = mc.getAgentPos()
            pitch = pos[3]
            visible = mc.getFullStat('LineOfSight')
            if visible and 'distance' in visible :
                dist = visible['distance']
                print(visible)
                return dist
            else:
                c += 1
                if c > 4:
                    return 0
                if pitch > 80:
                    prev_pitch = -0.05
                if pitch < -80:
                    prev_pitch = 0.05
                if prev_pitch is None:
                    prev_pitch = 0.05
                mc.sendCommand('pitch ' + str(prev_pitch))
                time.sleep(0.5)
                mc.sendCommand('pitch 0')
                continue 
        
def main():
    VereyaPython.setupLogger()
    unittest.main()

        
if __name__ == '__main__':
   main()
