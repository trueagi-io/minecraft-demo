from examples.log import setup_logger

import tagilmo.utils.mission_builder as mb
from tagilmo.utils.mathutils import *

from examples.vis import Visualizer

from examples.skills import *
from examples.agent import TAgent
from examples.minelogy import Minelogy

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
            self.blockMem.updateBlocks(self.rob)
            self.visualize()
        acts = self.goal.stop()
        for act in acts:
            self.rob.sendCommand(act)


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
    agent.rob.update_in_background()
    sleep(0.1)
    agent.rob.sendCommand("jump 0")

    #initialize_minelogy
    item_list = agent.rob.mc.getItemList()
    mlogy = Minelogy(item_list)

    '''
    Currently we don't use all recipes from the game since there are some issues with
    agent not be able to work properly with all recipes available
    '''
    # agent.rob.sendCommand('recipes')
    # sleep(2)
    # recipes = agent.rob.mc.observe[0]['recipes']
    # mlogy.set_recipes(recipes)

    agent.set_mlogy(mlogy)

    agent.run()
    visualizer.stop()
