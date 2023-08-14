import unittest
import logging
from tagilmo import VereyaPython
import json
import time
import tagilmo.utils.mission_builder as mb
from tagilmo.utils.vereya_wrapper import MCConnector, RobustObserver
from base_test import BaseTest
from common import init_mission


def getInvSafe(obs, item_type):
    inventory = obs.waitNotNoneObserve('getInventory', observeReq=False)
    return [item for item in inventory if item['type'] == item_type]


class TestCraft(unittest.TestCase):
    mc = None

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        start = (-108.0, 78, -187.0)
        mc, obs = init_mission(None, start_x=start[0], start_y=start[1], start_z=start[2], seed='5', forceReset='true')
        cls.mc = mc
        cls.obs = obs
        assert mc.safeStart()
        time.sleep(4)

    @classmethod
    def tearDownClass(cls, *args, **kwargs):
        cls.mc.stop()

    def setUp(self):
        super().setUp()
        self.mc.sendCommand("chat /clear")
        time.sleep(4)

    def test_swap_inventory(self):
        mc = self.mc
        obs = self.obs
        mc.sendCommand("chat /give @p minecraft:oak_planks 1")
        time.sleep(2)
        mc.sendCommand("chat /give @p wooden_pickaxe 1")
        time.sleep(2)
        
        pickaxe = getInvSafe(obs, 'wooden_pickaxe')
        oak_planks = getInvSafe(obs, 'wooden_pickaxe')
        print(oak_planks)
        print(pickaxe)
        pixidx = pickaxe[0]['index']
        oakidx = oak_planks[0]['index']

        mc.observeProc()
        inv1 = mc.getInventory()
        mc.sendCommand('swapInventoryItems {pixidx} {oakidx}')
        time.sleep(2)
        pickaxe = getInvSafe(obs, 'wooden_pickaxe')
        print(pickaxe)
        self.assertEquals(pickaxe[0]['index'], oakidx)

        
def main():
    unittest.main()

        
if __name__ == '__main__':
   main()
