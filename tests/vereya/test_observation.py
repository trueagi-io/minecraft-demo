import unittest
import logging
from tagilmo import VereyaPython
import json
import time
import tagilmo.utils.mission_builder as mb
from tagilmo.utils.vereya_wrapper import MCConnector, RobustObserver


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
             agentstart=mb.AgentStart([start_x, 78.0, start_y, 1]))])
    flat_json = {"biome":"minecraft:plains",
                 "layers":[{"block":"minecraft:diamond_block","height":1}],
                 "structures":{"structures": {"village":{}}}}

    flat_param = "3;7,25*1,3*3,2;1;stronghold,biome_1,village,decoration,dungeon,lake,mineshaft,lava_lake"
    flat_json = json.dumps(flat_json).replace('"', "%ESC")
    world = mb.defaultworld(
        seed='4',
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


class TestData(unittest.TestCase):
    mc = None
    rob = None

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        start = (-125.0, 71.0)
        mc, obs = init_mission(None, start_x=start[0], start_y=start[1]) 
        cls.mc = mc
        cls.rob = obs
        mc.safeStart()
        time.sleep(1)

    def setUp(self):
        time.sleep(2)

    def test_observation_from_ray(self):
        dist = self.getDist()
        self.assertGreater(dist, 0)

    def test_observation_from_chat(self):
        self.mc.sendCommand("chat get wooden_axe")
        command = self.rob.waitNotNoneObserve('getChat')
        self.assertEqual(command[0], "get wooden_axe")

    def test_observation_from_item(self):
        item_list, recipes = self.rob.mc.getItemAndRecipeList()
        self.assertGreater(len(item_list), 0, "item_list len")
        self.assertGreater(len(recipes), 0, "recipes len")

    def test_game_state(self):
        self.mc.observeProc()
        self.assertTrue(self.mc.getFullStat(key="isPaused") is not None)
        self.assertTrue(self.mc.getFullStat(key="input_type") is not None)

    def getDist(self):
        mc = self.mc
        c = 0
        while True:
            mc.observeProc()
            visible = mc.getFullStat('LineOfSight')
            if visible and 'distance' in visible :
                dist = visible['distance']
                print(visible)
                return dist
            else:
                c += 1
                if c > 4:
                    return 0
                mc.sendCommand('pitch 0.1')
                time.sleep(0.5)
                mc.sendCommand('pitch 0')
                continue 
        
def main():
    unittest.main()

        
if __name__ == '__main__':
   main()

