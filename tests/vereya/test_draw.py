import unittest
import logging
import json
import time
from math import floor
from tagilmo import VereyaPython
import tagilmo.utils.mission_builder as mb
from tagilmo.utils.vereya_wrapper import MCConnector, RobustObserver
from base_test import BaseTest
from common import count_items, init_mission

def test_basic_motion():
    pass 

class TestCraft(BaseTest):
    mc = None

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        start = (0, 0)
        cls.block = mb.DrawBlock((start[0] + 2), -60, start[1] + 2, "cobblestone")
        cls.cuboid = mb.DrawCuboid(start[0]-2, -61, start[1]-1, start[0]-2, -58, start[1]-2, "diamond_block")
        cls.line = mb.DrawLine(start[0] - 2, -62, start[1]-2, start[1] + 2, -62, start[1]+ 2, "redstone_block")
        cls.item = mb.DrawItem(start[0] + 1, -60, start[1], "diamond")
        draw = mb.DrawingDecorator([cls.cuboid, cls.line, cls.block, cls.item])
        mc, obs = init_mission(None, start_x=start[0], start_z=start[1], seed='4', forceReset="true", start_y=-60, worldType="flat", drawing_decorator=draw)
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

    def test_draw_block(self):
        mc = self.mc
        print('send ochat')
        grid = mc.getGridBox()
        grid_size = [grid[k][1] - grid[k][0] + 1 for k in range(3)]
        blocks = mc.getNearGrid()
        coords = mc.getAgentPos()
        lowest_point = [coords[k] + grid[k][0] for k in range(3)]
        block = self.block
        idx = int(block.x - lowest_point[0] + (block.z - lowest_point[2]) * grid_size[0] + (block.y - lowest_point[1]) * grid_size[0] * grid_size[2])
        real_block_type = blocks[idx]
        self.assertEqual(block.blockType, real_block_type)
        time.sleep(1)

    def test_draw_cuboid(self):
        mc = self.mc
        print('send ochat')
        grid = mc.getGridBox()
        grid_size = [grid[k][1] - grid[k][0] + 1 for k in range(3)]
        blocks = mc.getNearGrid()
        coords = mc.getAgentPos()
        lowest_point = [coords[k] + grid[k][0] for k in range(3)]
        cuboid = self.cuboid
        real_blocks = []
        x1, y1, z1 = cuboid.x1, cuboid.y1, cuboid.z1
        for y_ in range(cuboid.y2 - cuboid.y1 + 1):
            for z_ in range(cuboid.z2 - cuboid.z1 + 1):
                for x_ in range(cuboid.x2 - cuboid.x1 + 1):
                    idx = int(x1 + x_ - lowest_point[0] + (z1 + z_ - lowest_point[2]) * grid_size[0] + (y1 + y_ - lowest_point[1]) * grid_size[0] * grid_size[2])
                    real_blocks.append(blocks[idx]) 
        needed_blocks = [cuboid.blockType] * (cuboid.x2 - cuboid.x1 + 1) * (cuboid.y2 - cuboid.y1 + 1) * (cuboid.z2 - cuboid.z1 + 1)
        self.assertEqual(needed_blocks, real_blocks)
        time.sleep(1)

    def test_draw_line(self):
        mc = self.mc
        print('send ochat')
        grid = mc.getGridBox()
        grid_size = [grid[k][1] - grid[k][0] + 1 for k in range(3)]
        blocks = mc.getNearGrid()
        coords = mc.getAgentPos()
        lowest_point = [coords[k] + grid[k][0] for k in range(3)]
        line = self.line
        x1, y1, z1 = line.x1, line.y1, line.z1
        x2, y2, z2 = line.x2, line.y2, line.z2
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        steps = max(abs(dx), abs(dy), abs(dz))
        x_inc, y_inc, z_inc = dx / steps, dy / steps, dz / steps
        real_blocks = []
        x_, y_, z_ = 0, 0, 0
        for i in range(steps + 1):
            x_, y_, z_ = int(i * x_inc), int(i * y_inc), int(i * z_inc)
            idx = int(x1 + x_ - lowest_point[0] + (z1 + z_ - lowest_point[2]) * grid_size[0] + (y1 + y_ - lowest_point[1]) * grid_size[0] * grid_size[2])
            real_blocks.append(blocks[idx])
        needed_blocks = [line.blockType] * (steps + 1)
        self.assertEqual(needed_blocks, real_blocks)
        time.sleep(1)

    def test_draw_item(self):
        mc = self.mc
        print('send ochat')
        real_entity = mc.getNearEntities()[0] #there is only one entity in this test (agent does not count)
        entity = self.item
        real_entity_type = real_entity['name']
        #entities spawn at center of the block surface when summoned via /summon command
        self.assertEqual([entity.x, entity.y, entity.z], [floor(real_entity['x']), floor(real_entity['y']), floor(real_entity['z'])])
        self.assertEqual(entity.itemType, real_entity_type)
        time.sleep(1)

def main():
    VereyaPython.setupLogger()
    unittest.main()

        
if __name__ == '__main__':
   main()
