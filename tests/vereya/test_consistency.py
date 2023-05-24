import unittest
import logging
import json
import time
from tagilmo import VereyaPython
import tagilmo.utils.mission_builder as mb
from tagilmo.utils.vereya_wrapper import MCConnector, RobustObserver
from base_test import BaseTest


def init_mission(mc, start_x=None, start_y=None, start_z=None):
    want_depth = False
    video_producer = mb.VideoProducer(width=320 * 4,
                                      height=240 * 4, want_depth=want_depth)

    obs = mb.Observations()
    obs.gridNear = [[-1, 1], [-2, 1], [-1, 1]]


    agent_handlers = mb.AgentHandlers(observations=obs, video_producer=video_producer)

    print('starting at ({0}, {1})'.format(start_x, start_y))

    #miss = mb.MissionXML(namespace="ProjectMalmo.microsoft.com",
    miss = mb.MissionXML(
                         agentSections=[mb.AgentSection(name='Cristina',
             agenthandlers=agent_handlers,
                                      #    depth
             agentstart=mb.AgentStart([start_x, start_y, start_z, 1]))])
    world = mb.defaultworld(
        seed='96351635',
        forceReset="false",
        forceReuse="false")
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


def test_basic_motion():
    pass 


def count_items(inv, name):
    result = 0
    for elem in inv:
        if elem['type'] == name:
            result += elem['quantity']
    return result


class TestConsistency(BaseTest):
    mc = None

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        cls.start = (-72, 64, 28)
        mc, obs = init_mission(None, start_x=cls.start[0], start_y=cls.start[1], start_z=cls.start[2])
        cls.mc = mc
        cls.rob = obs
        assert mc.safeStart()
        time.sleep(4)

    @classmethod
    def tearDownClass(cls, *args, **kwargs):
        cls.mc.stop()

    def setUp(self):
        super().setUp()
        self.mc.sendCommand("chat /clear")
        time.sleep(4)

    def test_drop_and_see(self):
        mc = self.mc
        rob = self.rob
        blockdrops = rob.getBlocksDropsList()
        item_list, recipes = rob.getItemsAndRecipesLists()
        mc.sendCommand(f"chat /setblock -71 64 28 minecraft:oak_log")
        time.sleep(1)
        mc.sendCommand("chat /give @p oak_log 1")
        time.sleep(1)
        mc.observeProc()
        time.sleep(1)
        inv = mc.getInventory()
        oak_log_inventory_name = inv[0]['type']
        grid = mc.getNearGrid()
        equalityFlag = False
        for blockdrop in blockdrops:
            if blockdrop['block_name'] == "oak_log":
                self.assertEqual(blockdrop['item_name'], oak_log_inventory_name)
                for grid_item in grid:
                    if blockdrop['block_name'] == grid_item:
                        equalityFlag = True
                        break
                self.assertEqual(equalityFlag, True)
                break

        oaklog_in_itemlist = False
        for item in item_list:
            if item == oak_log_inventory_name:
                oaklog_in_itemlist = True
        self.assertEqual(oaklog_in_itemlist, True)
        oakplank_from_oaklog = False
        for recipe in recipes:
            if recipe['name'] == 'oak_planks':
                for ingr in recipe['ingredients'][0]:
                    if ingr['type'] == oak_log_inventory_name:
                        oakplank_from_oaklog = True
                        break
        self.assertEqual(oakplank_from_oaklog, True)
        mc.sendCommand("chat /loot spawn {} {} {} loot minecraft:blocks/oak_log".format(self.start[0], self.start[1]+2, self.start[2] + 3))
        time.sleep(1)
        rob.observeProcCached()
        time.sleep(1)
        nearEnt = rob.cached['getNearPickableEntities'][0]
        self.assertEqual(nearEnt[0]['name'], oak_log_inventory_name) #this test is currently failing due to different name comes from nearby entities

def main():
    VereyaPython.setupLogger()
    unittest.main()

        
if __name__ == '__main__':
   main()
