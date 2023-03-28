import os
import unittest
import logging
from tagilmo import VereyaPython
import json
import time
import tagilmo.utils.mission_builder as mb
from tagilmo.utils.vereya_wrapper import MCConnector, RobustObserver
from base_test import BaseTest


def init_mission(mc, start_x=None, start_y=None):
    want_depth = False
    video_producer = mb.VideoProducer(width=320 * 4,
                                      height=240 * 4, want_depth=want_depth)

    obs = mb.Observations()
    obs.gridNear = [[-1, 1], [-2, 1], [-1, 1]]


    agent_handlers = mb.AgentHandlers(observations=obs, video_producer=video_producer)

    print('starting at ({0}, {1})'.format(start_x, start_y))

    miss = mb.MissionXML(
                         agentSections=[mb.AgentSection(name='Cristina',
             agenthandlers=agent_handlers,
                                      #    depth
             agentstart=mb.AgentStart([start_x, 78.0, start_y, 1]))])
    flat_json = {"biome":"minecraft:plains",
                 "layers":[{"block":"minecraft:diamond_block","height":1}],
                 "structures":{"structures": {"village":{}}}}

    flat_param = "3;7,25*1,3*3,2;1;stronghold,biome_1,village,decoration,dungeon,lake,mineshaft,lava_lake"
    flat_json = json.dumps(flat_json).replace('"', "%ESC")
    world = mb.defaultworld(
        seed='4',
        forceReset="false",
        forceReuse="false")
    miss.setWorld(world)
    miss.serverSection.initial_conditions.allowedmobs = "Pig Sheep Cow Chicken Ozelot Rabbit Villager"
    # uncomment to disable passage of time:
    miss.serverSection.initial_conditions.time_pass = 'false'
    miss.serverSection.initial_conditions.time_start = "1000"

    if mc is None:
        mc = MCConnector(miss)
        mc.mission_record.is_recording_observations = True
        os.makedirs('test_observation', exist_ok=True)
        mc.mission_record.setDestination("test_observation")
        obs = RobustObserver(mc)
    else:
        mc.setMissionXML(miss)
    return mc, obs


logger = logging.getLogger(__name__)


class TestData(BaseTest):
    mc = None
    rob = None

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        start = (-125.0, 73.0)
        mc, obs = init_mission(None, start_x=start[0], start_y=start[1]) 
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
        blockdrops = self.rob.getBlocksDropsList()
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

    def test_game_state(self):
        self.mc.observeProc()
        self.assertTrue(self.mc.getFullStat(key="isPaused") is not None)
        self.assertTrue(self.mc.getFullStat(key="input_type") is not None)

    def getDist(self):
        mc = self.mc
        c = 0
        while True:
            mc.observeProc()
            visible = mc.getFullStat('LineOfSight')
            if visible and 'distance' in visible :
                dist = visible['distance']
                print(visible)
                return dist
            else:
                c += 1
                if c > 4:
                    return 0
                mc.sendCommand('pitch 0.1')
                time.sleep(0.5)
                mc.sendCommand('pitch 0')
                continue 
        
def main():
    unittest.main()

        
if __name__ == '__main__':
   main()

