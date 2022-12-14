import logging
from time import sleep, time
import math
from random import random

from tagilmo.utils.malmo_wrapper import MalmoConnector, RobustObserver
import tagilmo.utils.mission_builder as mb
from tagilmo.utils.mathutils import *

from examples.vis import Visualizer
from examples.Vereya import minelogy
from examples.Vereya.goal import *
from examples.Vereya.skills import *
from examples.Vereya.agent import TAgent

SCALE = 3

class Achiever(TAgent):

    def __init__(self, miss, visualizer=None, goal=None):
        super().__init__(miss, visualizer)
        self.set_goal(goal)

    def set_goal(self, goal=None):
        self.goal = ListenAndDo(self) if goal is None else goal

    def run(self):
        running = True
        while running:
            acts, running = self.goal.cycle()
            for act in acts:
                self.rob.sendCommand(act)
            sleep(0.05)
            self.rob.observeProcCached()
            self.blockMem.updateBlocks(self.rob)
            self.visualize()
        acts = self.goal.stop()
        for act in acts:
            self.rob.sendCommand(act)


def setup_logger():
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s: %(levelname)-8s %(message)s')
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(logging.DEBUG)
    # add ch to logger
    logger.addHandler(ch)


if __name__ == '__main__':
    setup_logger()
    visualizer = Visualizer()
    visualizer.start()
    video_producer = mb.VideoProducer(width=320 * SCALE, height=240 * SCALE, want_depth=False)
    agent_handlers = mb.AgentHandlers(video_producer=video_producer)
    miss = mb.MissionXML(agentSections=[mb.AgentSection(name='Robbo',
                                                        agenthandlers=agent_handlers)])

    world0 = mb.flatworld("3;7,25*1,3*3,2;1;stronghold,biome_1,village,decoration,dungeon,lake,mineshaft,lava_lake")
    world1 = mb.defaultworld(forceReset="false", forceReuse="true")
    miss.setWorld(world1)
    miss.serverSection.initial_conditions.allowedmobs = "Pig Sheep Cow Chicken Ozelot Rabbit Villager"

    agent = Achiever(miss, visualizer=visualizer)
    logging.info("Initializing the starting position")
    #those commands needed if we are reusing same world
    sleep(2)
    agent.rob.sendCommand("jump 1")
    sleep(0.1)
    agent.rob.sendCommand("jump 0")
    agent.run()
    visualizer.stop()
