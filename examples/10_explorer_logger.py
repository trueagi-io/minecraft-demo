from examples.log import setup_logger
from tagilmo import VereyaPython

from mcdemoaux.vision.vis import Visualizer
from examples.minelogy import Minelogy
from examples.skills import *
from mcdemoaux.logging.dataset_logger import *

from examples.knowledge_lists import *

from importlib import import_module

ex = import_module('7_explorer')


class ExplorerLogger(ex.Explorer):

    def __init__(self, mc, visualizer=None, goal=None, data_logger=None):
        super().__init__(mc, visualizer, goal)
        self.data_logger = data_logger

    def run(self):
        running = True
        while running:
            sleep(0.05)
            self.blockMem.updateBlocks(self.rob)
            self.kb.update()
            self.visualize()
            acts, running = self.goal.cycle()
            for act in acts:
                self.rob.sendCommand(act)
            self.data_logger.logImgActData(self.rob, acts)
        acts = self.goal.stop()
        for act in acts:
            self.rob.sendCommand(act)


if __name__ == '__main__':
    setup_logger()
    visualizer = Visualizer()
    visualizer.start()
    #seed 113 122 127? 128 129+? 130+? 131+?
    mc = MCConnector.connect(name='Robo', video=True, seed="151")
    data_logger = DatasetLogger()
    agent = ExplorerLogger(mc, visualizer=visualizer, data_logger=data_logger)
    sleep(4)

    # initialize minelogy
    item_list, recipes = agent.rob.getItemsAndRecipesLists()
    sleep(15)
    blockdrops = agent.rob.getBlocksDropsList()
    agent.rob.updatePassableBlocks()
    mlogy = Minelogy(item_list, items_to_craft, recipes, items_to_mine, blockdrops, ore_depths)
    agent.set_mlogy(mlogy)
    agent.run()

    '''
    rob = agent.rob
    skb = StaticKnowledge(rob)
    for i in range(600):
        sleep(0.2)
        rob.observeProcCached()
        skb.update()
        if skb.novelty_list != []:
            print(skb.novelty_list)
        skb.novelty_list = []
    '''

    visualizer.stop()
