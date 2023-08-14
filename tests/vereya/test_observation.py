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
        start = (-151.0, 78, -213.0)
        mc, obs = init_mission(None, start_x=start[0], start_y=start[1], start_z=start[2], forceReset='true', seed='43')
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

    def dist_test_function(self, visible):
        print(visible)
        dist = visible['distance']
        if 4.5 < dist:
            self.assertFalse(visible['inRange'])
        else:
            self.assertTrue(visible['inRange'])

    def _observation_from_ray(self, getter):
        time.sleep(1)
        visible = self.getDist(getter)
        dist = visible['distance']
        self.dist_test_function(visible)
        # random rotation
        self.mc.pitch((random.random() - 0.6)/10)
        self.mc.turn(random.random() - 0.5)
        time.sleep(1)
        self.mc.pitch(0)
        self.mc.turn(0)
        visible = self.getDist(getter)
        dist1 = visible['distance']
        self.dist_test_function(visible)
        self.assertNotEqual(dist, dist1, f"Distance {dist} != {dist1}")

    def test_observation_from_ray(self):
        return self._observation_from_ray(lambda: self.mc.getFullStat('LineOfSight'))

    def test_observation_from_ray_cached(self):
        return self._observation_from_ray(lambda: self.rob.getCachedObserve('getLineOfSights'))

    def test_observation_from_chat(self):
        logger.info("send chat ")
        self.mc.sendCommand("chat get wooden_axe")
        time.sleep(1)
        logger.info("wait chat")
        start = time.time()
        command = None
        while True:
            commands = self.rob.waitNotNoneObserve('getChat', observeReq=False)
            logger.debug(f'current chat {commands}')
            for c in commands:
                if c[0] is not None and "get wooden_axe" in c[0]:
                    command = c[0]
                    break
            time.sleep(0.05)
            end = time.time()
            if end - start > 2:
                break
        logger.info("result chat " + str(command))
        self.assertTrue(command is not None and "get wooden_axe" in command[0])

    def test_observation_from_item(self):
        logger.debug('getting items')
        item_list, recipes = self.rob.getItemsAndRecipesLists()
        logger.debug('got items %s', str(item_list))
        self.assertGreater(len(item_list), 0, "item_list len")
        self.assertTrue("diamond" in item_list)
        self.assertTrue("iron_ore" in item_list)
        self.assertTrue("stone_pickaxe" in item_list)
        self.assertGreater(len(recipes), 0, "recipes len")
        for recipe in recipes:
            if "iron_pickaxe" in recipe['name']:
                self.assertGreater(len(recipe['ingredients']), 0)
                for ingr in recipe['ingredients']:
                    if len(ingr) > 0:
                        self.assertTrue('iron_ingot' in ingr[0]['type'] or 'stick' in ingr[0]['type'])
            if "wooden_pickaxe" in recipe['name']:
                self.assertGreater(len(recipe['ingredients']), 0)
                for ingr in recipe['ingredients']:
                    if len(ingr) > 0:
                        self.assertTrue('stick' in ingr[0]['type'] or 'planks' in ingr[0]['type'])
            if recipe['name'] == "item.minecraft.stick":
                self.assertGreater(len(recipe['ingredients']), 0)
                for ingr in recipe['ingredients']:
                    if len(ingr) > 0:
                        self.assertTrue('planks' in ingr[0]['type'] or 'bamboo' in ingr[0]['type'])
            if recipe['name'] == "block.minecraft.furnace":
                self.assertGreater(len(recipe['ingredients']), 0)
                for ingr in recipe['ingredients']:
                    if len(ingr) > 0:
                        self.assertTrue('cobblestone' in ingr[0]['type'])


    def test_observation_from_triples(self):
        logger.debug('triples, getting blockdrops')
        blockdrops = self.rob.getBlocksDropsList()
        logger.debug('got blockdrops %s', str(blockdrops))
        self.assertGreater(len(blockdrops), 0, "blockdrops len")
        for blockdrop in blockdrops:
            if blockdrop['block_name'] == 'diamond_ore':
                tool, item = blockdrop['tool'], blockdrop['item_name']
                self.assertTrue((tool == 'iron_pickaxe' and item == "diamond") or
                                (item == 'diamond_ore' and tool == 'silkt_iron_pickaxe'))
            if blockdrop['block_name'] == 'iron_ore':
                tool, item = blockdrop['tool'], blockdrop['item_name']
                self.assertTrue((tool == 'stone_pickaxe' and item == "raw_iron") or
                                (item == 'iron_ore' and tool == 'silkt_stone_pickaxe'))
            if blockdrop['block_name'] == 'birch_leaves' and blockdrop['tool'] == 'shears':
                self.assertEqual(blockdrop['item_name'], 'birch_leaves', "check leaves")

    def test_observation_from_find_block(self):
        logger.debug('finding block in big grid')
        agentpos = self.rob.getCachedObserve('getAgentPos')
        diamond_x = int(agentpos[0] + random.randint(1, 10))
        diamond_y = int(agentpos[1] + random.randint(1, 10))
        diamond_z = int(agentpos[2] + random.randint(1, 10))
        self.mc.sendCommand("chat /setblock {} {} {} minecraft:diamond_ore".format(diamond_x, diamond_y, diamond_z))
        time.sleep(1)
        self.rob.sendCommandToFindBlock("diamond_ore")
        time.sleep(1)
        diamond_ore_loc = None
        bigGridObservations = self.rob.waitNotNoneObserve('getBlockFromBigGrid')
        for obs in reversed(bigGridObservations):
            if obs[0] is not None:
                diamond_ore_loc = obs[0]
                break
        self.assertEqual(diamond_ore_loc[0], diamond_x)
        self.assertEqual(diamond_ore_loc[1], diamond_y)
        self.assertEqual(diamond_ore_loc[2], diamond_z)
        self.assertEqual(diamond_ore_loc[3], 'diamond_ore')
        self.mc.sendCommand("chat /setblock {} {} {} minecraft:air".format(diamond_x, diamond_y, diamond_z))

    def test_game_state(self):
        for i in range(10):
            if self.mc.getFullStat(key="isPaused") is not None:
                break
            time.sleep(0.5)
        self.assertTrue(self.mc.getFullStat(key="isPaused") is not None)
        self.assertTrue(self.mc.getFullStat(key="input_type") is not None)

    def getDist(self, getter):
        mc = self.mc
        c = 0
        prev_pitch = None
        while True:
            pos = mc.getAgentPos()
            pitch = pos[3]
            visible = getter() 
            if visible and 'distance' in visible :
                print(visible)
                return visible
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
