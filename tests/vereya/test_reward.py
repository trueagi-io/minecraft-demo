import unittest
import warnings
import os
import json
import time
from tagilmo import VereyaPython
import tagilmo.utils.mission_builder as mb
from tagilmo.utils.vereya_wrapper import MCConnector, RobustObserver
from base_test import BaseTest

def init_mission(mc, start_x, start_z, seed, forceReset="false",
                 forceReuse="false", start_y=78, worldType = "default", 
                 drawing_decorator=None, rewardForSendingCommand=None, rewardForTouchingBlockType=None,
                 serverIp=None, serverPort=0):


    want_depth = False
    video_producer = mb.VideoProducer(width=320 * 4,
                                      height=240 * 4, want_depth=want_depth)

    obs = mb.Observations()
    obs.gridNear = [[-2, 2], [-2, 2], [-2, 2]]


    agent_handlers = mb.AgentHandlers(observations=obs, video_producer=video_producer, 
                                      rewardForSendingCommand=rewardForSendingCommand,
                                      rewardForTouchingBlockType=rewardForTouchingBlockType)

    print('starting at ({0}, {1})'.format(start_x, start_y))

    start = [start_x, start_y, start_z, 1]
    if all(x is None for x in [start_x, start_y, start_z]):
        start = None
    miss = mb.MissionXML(
                    agentSections=[mb.AgentSection(name='Cristina',
                        agenthandlers=agent_handlers,
                        agentstart=mb.AgentStart(start))],
                    serverSection=mb.ServerSection(handlers=mb.ServerHandlers(drawingdecorator=drawing_decorator)))
    flat_json = {"biome":"minecraft:plains",
                 "layers":[{"block":"minecraft:diamond_block","height":1}],
                 "structures":{"structures": {"village":{}}}}

    flat_param = "3;7,25*1,3*3,2;1;stronghold,biome_1,village,decoration,dungeon,lake,mineshaft,lava_lake"
    flat_json = json.dumps(flat_json).replace('"', "%ESC")
    match worldType:
        case "default":
            world = mb.defaultworld(
                seed=seed,
                forceReset=forceReset,
                forceReuse=forceReuse)
        case "flat":
            world = mb.flatworld("",
                                seed=seed,
                                forceReset=forceReset)
        case _:
            warnings.warn("World type " + worldType + " is not supported, setting up default world")
            world = mb.defaultworld(
                seed=seed,
                forceReset=forceReset,
                forceReuse=forceReuse)
    miss.setWorld(world)
    miss.serverSection.initial_conditions.allowedmobs = "Pig Sheep Cow Chicken Ozelot Rabbit Villager"
    # uncomment to disable passage of time:
    miss.serverSection.initial_conditions.time_pass = 'false'
    miss.serverSection.initial_conditions.time_start = "1000"
    if not os.path.exists('./observations'):
        os.mkdir('./observations')

    if mc is None:
        mc = MCConnector(miss, serverIp=serverIp, serverPort=serverPort)
        mc.mission_record.setDestination('./observations/')
        mc.mission_record.is_recording_observations = True
        obs = RobustObserver(mc)
    else:
        mc.setMissionXML(miss)
    return mc, obs


class TestReward(BaseTest):
    mc = None

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        start = (0, 0)
        cls.rewardForSending = mb.RewardForSendingCommand(-1.0)
        cls.rewardBlock = mb.Block(reward=100.0, blockType="stone", behaviour="onceOnly")
        rewardForTouching = mb.RewardForTouchingBlockType(rewardBlocks=[cls.rewardBlock])
        cls.block = mb.DrawBlock((start[0] + 1), -61, start[1], "stone")
        draw = mb.DrawingDecorator([cls.block])
        mc, obs = init_mission(None, start_x=start[0], start_z=start[1], seed='4', 
                               forceReset="true", start_y=-60, worldType="flat", 
                               drawing_decorator=draw,rewardForSendingCommand=cls.rewardForSending, 
                               rewardForTouchingBlockType=rewardForTouching)
        cls.mc = mc
        cls.obs = obs
        assert mc.safeStart()
        time.sleep(4)

    @classmethod
    def tearDownClass(cls, *args, **kwargs):
        cls.mc.stop()

    def setUp(self):
        super().setUp()
        time.sleep(4)

    def test_reward_for_sending_command(self):
        mc = self.mc
        #check accumulated rewards for several commands 
        turn_rate = 0.1
        turn_num = 5
        for _ in range(turn_num):
            mc.turn(turn_rate)
            time.sleep(0.5)
            mc.turn(0)
            time.sleep(0.5)
            turn_rate *= -1
        mc.discreteMove("west")
        time.sleep(0.5)
        mc.discreteMove("east")
        time.sleep(0.5)
        cum_reward = mc.getRewards()
        assert len(cum_reward) > 0
        cum_reward = cum_reward[0].getAsSimpleString()
        cum_reward = cum_reward.split(": ")[1]
        theor_reward = str(-1. *(turn_num * 2 + 2))
        self.assertEqual(cum_reward, theor_reward)
        time.sleep(1)
        
    def test_reward_for_touching_block(self):
        mc = self.mc
        mc.discreteMove("east")
        time.sleep(0.5)
        #we gonna get a sum of rewards for moving (-1) and touching stone (100)
        cum_reward = mc.getRewards()
        assert len(cum_reward) > 0
        cum_reward = cum_reward[0].getAsSimpleString()
        cum_reward = cum_reward.split(": ")[1]
        theor_reward = str(self.rewardForSending.reward + self.rewardBlock.reward)
        self.assertEqual(theor_reward, cum_reward)


def main():
    VereyaPython.setupLogger()
    unittest.main()

        
if __name__ == '__main__':
   main()
