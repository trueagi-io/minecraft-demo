from base_test import BaseTest
from common import init_mission
import unittest
import time


class TestPlacement(BaseTest):
    mc = None

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        start = (-125.0, 73)
        mc, obs = init_mission(None, start_x=start[0], start_z=start[1], seed=4,
                               forceReset='true')
        cls.mc = mc
        cls.rob = obs
        mc.safeStart()
        time.sleep(1)

    @classmethod
    def tearDownClass(cls, *args, **kwargs):
        cls.mc.stop()

    def test_placement(self):
        """ test block placement api"""
        mc = self.mc
        rob = self.rob
        time.sleep(1)
        grid = rob.getCachedObserve('getNearGrid')
        pos = rob.getCachedObserve('getAgentPos')
        x, y, z = [int(_) for _ in pos[:3]]
        mc.sendCommand(f"placeBlock {x} {y + 2} {z} minecraft:dirt replace")
        time.sleep(1)
        grid1 = rob.getCachedObserve('getNearGrid')
        self.assertNotEqual(grid, grid1, "Grids are equal")

if __name__ == "__main__":
    unittest.main()
