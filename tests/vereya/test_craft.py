import unittest
import logging
import json
import time
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
        start = (-125.0, 78, 71.0)
        mc, obs = init_mission(None, start_x=start[0], start_y=start[1], start_z=start[2], seed='4', forceReset="true")
        cls.mc = mc
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
        print('send ochat')
        mc.sendCommand("chat /give @p oak_planks 1")
        time.sleep(1)
        mc.sendCommand("chat /give @p birch_planks 1")
        time.sleep(1)
        

    def test_cook_fish(self):
        pass
    
    def test_craft_stick(self):
        mc = self.mc
        mc.observeProc()
        inv = mc.getInventory()
        # result is list of dicts
        print('send ochat')
        mc.sendCommand("chat /give @p oak_planks 1")
        time.sleep(1)
        mc.sendCommand("chat /give @p birch_planks 1")
        time.sleep(1)
        mc.sendCommand("craft stick")
        time.sleep(2)
        mc.observeProc()
        inv1 = mc.getInventory()
        self.assertEqual(count_items(inv, "stick") + 4, count_items(inv1, "stick"))
        self.assertEqual(count_items(inv1, "birch_planks"), 0)
        print('sending command')

    def test_craft_pickaxe(self):
        mc = self.mc
        mc.sendCommand("chat /give @p oak_planks 1")
        time.sleep(1)
        mc.sendCommand("chat /give @p birch_planks 1")
        time.sleep(1)
        mc.sendCommand("chat /give @p acacia_planks 1")
        time.sleep(1)
        mc.sendCommand("chat /give @p stick 2")
        time.sleep(1)
        mc.sendCommand("chat /give @p crafting_table 1")
        time.sleep(1)
        mc.observeProc()
        time.sleep(1)
        print('making pickaxe')
        inv = mc.getInventory()
        mc.sendCommand("craft wooden_pickaxe")
        time.sleep(2)
        mc.observeProc()
        inv1 = mc.getInventory()
        self.assertEqual(count_items(inv, "wooden_pickaxe") + 1, count_items(inv1, "wooden_pickaxe"))
        self.assertEqual(count_items(inv, "acacia_planks") - 1, count_items(inv1, "acacia_planks"))

    def test_smoking_rabbit(self):
        mc = self.mc
        time.sleep(1)
        mc.sendCommand("chat /give @p rabbit 1")
        time.sleep(1)
        mc.sendCommand("chat /give @p coal 1")
        time.sleep(1)
        mc.sendCommand("chat /give @p furnace 1")
        time.sleep(1)
        mc.observeProc()
        time.sleep(1)
        print('cooking rabbit using furnace')
        inv = mc.getInventory()
        mc.sendCommand("craft cooked_rabbit coal")
        time.sleep(2)
        mc.observeProc()
        time.sleep(1)
        inv1 = mc.getInventory()
        self.assertEqual(count_items(inv, "cooked_rabbit") + 1, count_items(inv1, "cooked_rabbit"))
        self.mc.sendCommand("chat /clear")
        time.sleep(2)
        mc.sendCommand("chat /give @p rabbit 1")
        time.sleep(1)
        mc.sendCommand("chat /give @p campfire 1")
        time.sleep(1)
        mc.observeProc()
        time.sleep(1)
        print('cooking rabbit using campfire')
        inv = mc.getInventory()
        mc.sendCommand("craft cooked_rabbit")
        time.sleep(2)
        mc.observeProc()
        time.sleep(1)
        inv1 = mc.getInventory()
        self.assertEqual(count_items(inv, "cooked_rabbit") + 1, count_items(inv1, "cooked_rabbit"))
        #self.mc.sendCommand("chat /clear")
        #time.sleep(2)
        #mc.sendCommand("chat /give @p rabbit 1")
        #time.sleep(1)
        #mc.sendCommand("chat /give @p coal 1")
        #time.sleep(1)
        #mc.sendCommand("chat /give @p smoker 1")
        #time.sleep(1)
        #mc.observeProc()
        #time.sleep(1)
        #print('cooking rabbit using smoker')
        #inv = mc.getInventory()
        #mc.sendCommand("craft cooked_rabbit coal") # easy craft not working with smoker
        #time.sleep(2)
        #mc.observeProc()
        #time.sleep(1)
        #inv1 = mc.getInventory()
        #self.assertEqual(count_items(inv, "cooked_rabbit") + 1, count_items(inv1, "cooked_rabbit"))

    def test_blasting(self):
        mc = self.mc
        time.sleep(1)
        mc.sendCommand("chat /give @p deepslate_iron_ore 1")
        time.sleep(1)
        mc.sendCommand("chat /give @p blast_furnace 1")
        time.sleep(1)
        mc.sendCommand("chat /give @p coal 1")
        time.sleep(1)
        mc.observeProc()
        time.sleep(1)
        print('blasting iron_ore')
        inv = mc.getInventory()
        mc.sendCommand("craft iron_ingot coal")
        time.sleep(2)
        mc.observeProc()
        time.sleep(1)
        inv1 = mc.getInventory()
        self.assertEqual(count_items(inv, "iron_ingot") + 1, count_items(inv1, "iron_ingot"))

    #def test_stonecutter(self):
        #mc = self.mc
        #time.sleep(1)
        #mc.sendCommand("chat /give @p andesite 1")
        #time.sleep(1)
        #mc.sendCommand("chat /give @p stonecutter 1")
        #time.sleep(1)
        #mc.observeProc()
        #time.sleep(1)
        #print('getting wall using stonecutter')
        #inv = mc.getInventory()
        #mc.sendCommand("craft andesite_wall")  # easy craft not working with stonecutter
        #time.sleep(2)
        #mc.observeProc()
        #time.sleep(1)
        #inv1 = mc.getInventory()
        #self.assertEqual(count_items(inv, "andesite_wall") + 1, count_items(inv1, "andesite_wall"))

    def test_failed(self):
        mc = self.mc
        time.sleep(1)
        mc.sendCommand("chat /give @p oak_planks 1")
        time.sleep(1)
        mc.sendCommand("chat /give @p birch_planks 1")
        time.sleep(1)
        mc.sendCommand("chat /give @p acacia_planks 1")
        time.sleep(1)
        mc.sendCommand("chat /give @p stick 1")
        time.sleep(1)
        mc.observeProc()
        time.sleep(1)
        print('making pickaxe')
        inv = mc.getInventory()
        mc.sendCommand("craft wooden_pickaxe")
        time.sleep(2)
        mc.observeProc()
        time.sleep(1)
        inv1 = mc.getInventory()
        self.assertEqual(count_items(inv, "wooden_pickaxe"), count_items(inv1, "wooden_pickaxe"))
        self.assertEqual(count_items(inv, "acacia_planks"), count_items(inv1, "acacia_planks"))
        self.assertEqual(count_items(inv, "stick"), count_items(inv1, "stick"))

    def test_smelt_iron(self):
        mc = self.mc
        mc.observeProc()
        inv = mc.getInventory()
        # result is list of dicts
        print('send ochat')
        mc.sendCommand("chat /give @p birch_planks 1")
        time.sleep(1)
        mc.sendCommand("chat /give @p raw_iron 1")
        time.sleep(1)
        mc.sendCommand("craft iron_ingot birch_planks")
        time.sleep(2)
        mc.observeProc()
        inv1 = mc.getInventory()
        self.assertEqual(count_items(inv1, "iron_ingot"), 0)
        self.assertEqual(count_items(inv1, "raw_iron"), 1)
        self.assertEqual(count_items(inv1, "birch_planks"), 1)

        mc.sendCommand("chat /give @p furnace")
        time.sleep(1)
        mc.sendCommand("craft iron_ingot birch_planks")
        time.sleep(2)

        mc.observeProc()
        inv1 = mc.getInventory()
        self.assertEqual(count_items(inv1, "iron_ingot"), 1)
        self.assertEqual(count_items(inv1, "raw_iron"), 0)
        self.assertEqual(count_items(inv1, "birch_planks"), 0)
        print('sending command')

       


def main():
    VereyaPython.setupLogger()
    unittest.main()
#    VereyaPython.setLoggingComponent(VereyaPython.LoggingComponent.LOG_TCP, True)
#    VereyaPython.setLogging('log.txt', VereyaPython.LoggingSeverityLevel.LOG_FINE)

        
if __name__ == '__main__':
   main()
