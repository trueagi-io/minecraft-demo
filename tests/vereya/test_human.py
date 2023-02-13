import unittest
import logging
from tagilmo import VereyaPython
import json
import time
import tagilmo.utils.mission_builder as mb
from tagilmo.utils.vereya_wrapper import MCConnector, RobustObserver


def init_mission(mc, start_x=None, start_y=None):
    want_depth = False
    # quit by reaching target or when zero health
    mission_ending = """
        <MissionQuitCommands quitDescription="give_up"/>
        <RewardForMissionEnd>
          <Reward description="give_up" reward="243"/>
        </RewardForMissionEnd>
        """

    video_producer = mb.VideoProducer(width=320 * 4,
                                      height=240 * 4, want_depth=want_depth)

    obs = mb.Observations(bHuman=False)
    obs = mb.Observations(bHuman=True)
    obs.gridNear = [[-1, 1], [-2, 1], [-1, 1]]


    agent_handlers = mb.AgentHandlers(observations=obs,
            all_str=mission_ending, video_producer=video_producer)

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


class TestQuit(unittest.TestCase):
    mc: MCConnector = None

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        start = 316.5, 5375.5
        start = (-108.0, -187.0)
        mc, obs = init_mission(None, start_x=start[0], start_y=start[1]) 
        cls.mc = mc
        assert mc.safeStart()
        time.sleep(3)

    @classmethod
    def tearDownClass(cls, *args, **kwargs):
        cls.mc.stop()

    def setUp(self):
        self.mc.sendCommand("chat /clear")
        time.sleep(3)

    def test_human(self):
        self.assertTrue(self.mc.is_mission_running())
        print('send ochat')
        while True:
            time.sleep(0.02)
            self.mc.observeProc()
            if self.mc.observe[0] is None:
                continue
            obs = self.mc.observe[0].get("input_events", None)
            if obs is not None:
                print(obs)
        self.mc.sendCommand("quit")
        time.sleep(4)
        self.assertFalse(self.mc.is_mission_running())


def main():
    VereyaPython.setupLogger()
    unittest.main()
        
if __name__ == '__main__':
   main()
