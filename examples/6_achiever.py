import torch
import logging
from time import sleep, time
from tagilmo.utils.malmo_wrapper import MalmoConnector, RobustObserver
import tagilmo.utils.mission_builder as mb
from tagilmo.utils.mathutils import *
from vis import Visualizer
import math
from random import random

from examples import minelogy
from examples.goal import *
from examples.skills import *
from examples.agent import TAgent

SCALE = 3

class ListenAndDo(Switcher):

    def __init__(self, agent):
        super().__init__(agent.rob)
        self.agent = agent
        self.next_command = None
        self.terminate = False

    def update(self):
        self.next_command = self.rob.cached['getChat'][0]
        if self.next_command is not None:
            words = self.next_command[0].split(' ')
            self.rob.cached['getChat'] = (None, self.rob.cached['getChat'][1])
            if self.delegate is not None:
                self.stopDelegate = True
            else:
                print("Receive command: ", self.next_command)
                if words[-1] == 'terminate':
                    self.terminate = True
                if words[-2] == 'get':
                    self.delegate = Obtain(self.agent, [{'type': words[-1]}])
        super().update()

    def finished(self):
        return self.terminate


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
                agenthandlers=agent_handlers,)])

    world0 = mb.flatworld("3;7,25*1,3*3,2;1;stronghold,biome_1,village,decoration,dungeon,lake,mineshaft,lava_lake")
    world1 = mb.defaultworld(forceReset="true")
    miss.setWorld(world1)

    agent = Achiever(miss, visualizer=visualizer)
    agent.run()
    visualizer.stop()
