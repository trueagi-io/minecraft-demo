from base_test import BaseTest
from common import init_mission
import unittest
import time


class TestPlacement(BaseTest):
    mc = None

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        start = (-125.0, 73.0)
        mc, obs = init_mission(None, start_x=start[0], start_y=start[1], seed=4)
        cls.mc = mc
        cls.rob = obs
        mc.safeStart()
        time.sleep(1)

    def test_placement(self):
        """ test block placement api"""
        mc = self.mc
        rob = self.rob
        rob.update_in_background()

        time.sleep(1)
        grid = rob.getCachedObserve('getNearGrid')
        mc.sendCommand("placeBlock -125 72 74 minecraft:dirt replace")
        time.sleep(1)
        grid1 = rob.getCachedObserve('getNearGrid')
        self.assertNotEqual(grid, grid1, "Grids are equal")

if __name__ == "__main__":
    unittest.main()