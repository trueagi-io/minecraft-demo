import unittest
import logging
import json
import time
from tagilmo import VereyaPython
import tagilmo.utils.mission_builder as mb
from tagilmo.utils.malmo_wrapper import MalmoConnector, RobustObserver
from experiments import common
from experiments.common import stop_motion, grid_to_vec_walking, direction_to_target



def init_mission(mc, start_x=None, start_y=None):
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
             agentstart=mb.AgentStart([start_x, 74.0, start_y, 1]))])
    flat_json = {"biome":"minecraft:plains",
                 "layers":[{"block":"minecraft:diamond_block","height":1}],
                 "structures":{"structures": {"village":{}}}}

    flat_param = "3;7,25*1,3*3,2;1;stronghold,biome_1,village,decoration,dungeon,lake,mineshaft,lava_lake"
    flat_json = json.dumps(flat_json).replace('"', "%ESC")
    world = mb.defaultworld(
        seed='5',
        forceReset="false",
        forceReuse="true")
    miss.setWorld(world)
    miss.serverSection.initial_conditions.allowedmobs = "Pig Sheep Cow Chicken Ozelot Rabbit Villager"
    # uncomment to disable passage of time:
    miss.serverSection.initial_conditions.time_pass = 'false'
    miss.serverSection.initial_conditions.time_start = "1000"

    if mc is None:
        mc = MalmoConnector(miss)
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


class TestCraft(unittest.TestCase):
    mc = None

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        start = 316.5, 5375.5
        start = (-108.0, -187.0)
        mc, obs = init_mission(None, start_x=start[0], start_y=start[1]) 
        cls.mc = mc
        mc.safeStart()
        time.sleep(4)

    def setUp(self):
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
