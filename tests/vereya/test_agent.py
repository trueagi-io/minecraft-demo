import unittest
import logging
import json
import time
from tagilmo import VereyaPython
import tagilmo.utils.mission_builder as mb
from tagilmo.utils.vereya_wrapper import MCConnector, RobustObserver
from base_test import BaseTest
from common import init_mission, count_items
from mcdemoaux.agenttools.agent import TAgent
from examples.minelogy import Minelogy
from examples.knowledge_lists import *
import examples.skills


item_to_obtain = "stone_pickaxe"

class Tester(TAgent):

    def __init__(self, mc, visualizer=None, goal=None):
        super().__init__(mc, visualizer)
        self.set_goal(goal)

    def set_goal(self, goal=None):
        self.goal = examples.skills.Obtain(self, [item_to_obtain])

    def run(self):
        running = True
        while running:
            acts, running = self.goal.cycle()
            for act in acts:
                self.rob.sendCommand(act)
            time.sleep(0.05)
            self.blockMem.updateBlocks(self.rob)
        acts = self.goal.stop()
        for act in acts:
            self.rob.sendCommand(act)


class TestAgent(BaseTest):
    mc = None

    @classmethod
    def setUpClass(self, *args, **kwargs):
        start = (4.0, 69.0, 68)
        mc, obs = init_mission(None, start_x=start[0], start_y=start[1], start_z=start[2], forceReset='true', seed='2')
        self.mc = mc
        self.rob = obs
        assert mc.safeStart()
        time.sleep(4)

    @classmethod
    def tearDownClass(self, *args, **kwargs):
        self.mc.stop()

    def setUp(self):
        super().setUp()
        self.mc.sendCommand("chat /clear")
        time.sleep(4)

    def test_agent(self):
        mc = self.mc
        agent = Tester(mc)

        # initialize_minelogy
        item_list, recipes = agent.rob.getItemsAndRecipesLists()
        self.assertNotEqual(item_list, None, "check item_list not None")
        self.assertNotEqual(recipes, None, "check recipes not None")
        blockdrops = agent.rob.getBlocksDropsList()
        self.assertNotEqual(blockdrops, None, "check blockdrops not None")
        agent.rob.updatePassableBlocks()
        try:
            mlogy = Minelogy(item_list, items_to_craft, recipes, items_to_mine, blockdrops, ore_depths)
        except Exception as e:
            print(f'Exception occured: {e}')
            return
        agent.set_mlogy(mlogy)
        agent.run()
        mc.observeProc()
        inv = mc.getInventory()
        self.assertEqual(count_items(inv, item_to_obtain), 1, msg=f"check if {item_to_obtain} was crafted")


class TestAgentServer(TestAgent):
    @classmethod
    def setUpClass(self, *args, **kwargs):
        start = (4.0, 69.0, 68)
        mc, obs = init_mission(None, start_x=None, start_y=None, start_z=None, forceReset='false', forceReuse=True, seed='2', serverIp='127.0.0.1', serverPort=25565)
        self.mc = mc
        self.rob = obs
        assert mc.safeStart()
        time.sleep(4)


def main():
    VereyaPython.setupLogger()
    unittest.main()


if __name__ == '__main__':
    main()
