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


def getInvSafe(obs, item_type):
    inventory = obs.waitNotNoneObserve('getInventory', observeReq=False)
    return [item for item in inventory if item['type'] == item_type]


class TestCraft(unittest.TestCase):
    mc = None

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        start = 316.5, 5375.5
        start = (-108.0, -187.0)
        mc, obs = init_mission(None, start_x=start[0], start_y=start[1]) 
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
        print(pickaxe)

        mc.observeProc()
        inv1 = mc.getInventory()
        mc.sendCommand('swapInventoryItems 0 ' + str(pickaxe[0]['index']))
        time.sleep(2)
        pickaxe = getInvSafe(obs, 'wooden_pickaxe')
        self.assertEquals(pickaxe[0]['index'], 0)

        
def main():
    unittest.main()

        
if __name__ == '__main__':
   main()
