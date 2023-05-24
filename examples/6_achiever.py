from examples.log import setup_logger

from mcdemoaux.vision.vis import Visualizer

from examples.skills import *
from mcdemoaux.agenttools.agent import TAgent
from examples.minelogy import Minelogy
from examples.knowledge_lists import *

class Achiever(TAgent):

    def __init__(self, mc, visualizer=None, goal=None):
        super().__init__(mc, visualizer)
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
    mc = MCConnector.connect(name='Robbo', video=True)
    agent = Achiever(mc, visualizer=visualizer)
    logging.info("Initializing the starting position")
    #those commands needed if we are reusing same world
    sleep(2)
    agent.rob.sendCommand("jump 1")
    agent.rob.update_in_background()
    sleep(0.1)
    agent.rob.sendCommand("jump 0")

    #initialize_minelogy
    item_list, recipes = agent.rob.getItemsAndRecipesLists()
    blockdrops = agent.rob.getBlocksDropsList()
    mlogy = Minelogy(item_list, items_to_craft, recipes, items_to_mine, blockdrops, ore_depths)
    '''
    Currently we don't use all recipes from the game since there are some issues with
    agent not be able to work properly with all recipes available
    '''
    agent.set_mlogy(mlogy)

    agent.run()
    visualizer.stop()
