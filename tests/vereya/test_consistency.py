import unittest
import logging
import json
import time
from tagilmo import VereyaPython
import tagilmo.utils.mission_builder as mb
from tagilmo.utils.vereya_wrapper import MCConnector, RobustObserver
from base_test import BaseTest
from common import init_mission


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
        start = (-151.0, -213.0)
        mc, obs = init_mission(None, start_x=start[0], start_y=start[1], forceReset='true', seed='43')
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
        agentpos = self.rob.getCachedObserve('getAgentPos')
        log_x = int(agentpos[0] + 1)
        log_y = int(agentpos[1])
        log_z = int(agentpos[2])
        self.mc.sendCommand(f"chat /setblock {log_x} {log_y} {log_z} minecraft:oak_log")
        blockdrops = rob.getBlocksDropsList()
        item_list, recipes = rob.getItemsAndRecipesLists()
        mc.sendCommand("chat /give @p oak_log 1")
        time.sleep(1)
        mc.observeProc()
        time.sleep(1)
        inv = mc.getInventory()
        oak_log_inventory_name = inv[0]['type']
        grid = mc.getNearGrid()
        oak_log_world_block_name = ""
        for grid_item in grid:
            if "oak_log" in grid_item:
                oak_log_world_block_name = grid_item
                break
        for blockdrop in blockdrops:
            if blockdrop['block_name'] == "oak_log":
                self.assertEqual(blockdrop['item_name'], oak_log_inventory_name)
                self.assertEqual(blockdrop['block_name'], oak_log_world_block_name)
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
        mc.sendCommand(
            "chat /loot spawn {} {} {} loot minecraft:blocks/oak_log".format(agentpos[0], agentpos[1] + 2,
                                                                             agentpos[2] + 3))
        time.sleep(1)
        rob.observeProcCached()
        time.sleep(1)
        nearEnt = rob.cached['getNearPickableEntities'][0]
        self.assertEqual(nearEnt[0]['name'],
                         oak_log_inventory_name)  # this test is currently failing due to different name comes from nearby entities


def main():
    VereyaPython.setupLogger()
    unittest.main()


if __name__ == '__main__':
    main()
