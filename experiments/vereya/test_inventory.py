import unittest
import logging
from tagilmo import VereyaPython
import json
import time
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


def getInvSafe(mc, item):
    while True:
        time.sleep(0.3)
        mc.observeProc()
        if mc.isInventoryAvailable(): return RobustObserver(mc).filterInventoryItem(item)


class TestCraft(unittest.TestCase):
    mc = None

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        start = 316.5, 5375.5
        start = (-108.0, -187.0)
        mc, obs = init_mission(None, start_x=start[0], start_y=start[1]) 
        cls.mc = mc
        assert mc.safeStart()
        time.sleep(4)

    @classmethod
    def tearDownClass(cls, *args, **kwargs):
        cls.mc.stop()

    def setUp(self):
        self.mc.sendCommand("chat /clear")
        time.sleep(4)

    def test_swap_inventory(self):
        mc = self.mc
        mc.sendCommand("chat /give @p minecraft:oak_planks 1")
        import pdb;pdb.set_trace()
        time.sleep(1)
        mc.sendCommand("chat /give @p wooden_pickaxe 1")
        time.sleep(1)
        
        pickaxe = getInvSafe(mc, 'wooden_pickaxe')

        mc.observeProc()
        inv1 = mc.getInventory()
        mc.sendCommand('swapInventoryItems 0 ' + str(pickaxe[0]['index']))
        time.sleep(1)
        pickaxe = getInvSafe(mc, 'wooden_pickaxe')
        self.assertEquals(pickaxe[0]['index'], 0)

        
def main():
    unittest.main()
#    VereyaPython.setLoggingComponent(VereyaPython.LoggingComponent.LOG_TCP, True)
#    VereyaPython.setLogging('log.txt', VereyaPython.LoggingSeverityLevel.LOG_FINE)

        
if __name__ == '__main__':
   main()
