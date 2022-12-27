import logging
from experiments.log import setup_logger
from time import sleep, time
from tagilmo.utils.malmo_wrapper import MalmoConnector, RobustObserver
import tagilmo.utils.mission_builder as mb
from tagilmo.utils.mathutils import *
import math
from random import random

from examples.vis import Visualizer
from examples import minelogy
from examples.goal import *
from examples.skills import *
from examples.agent import TAgent

class StaticKnowledge:

    ID_PARAMS = ['source', 'type', 'name', 'hitType', 'variant'] # + prop_type?

    def __init__(self, rob):
        self.rob = rob
        self.kb = {}
        self.novelty_list = []

    def update(self):
        sources = ['getInventory', 'getNearEntities', 'getLineOfSights', 'getNearGrid']
        for source in sources:
            data = self.rob.cached[source][0]
            tm = self.rob.cached[source][1]
            if data is None:
                continue
            if isinstance(data, dict):
                data = [data]
            for entry in data:
                self.update_entry(
                    dict(entry if isinstance(entry, dict) else {'type': entry}, **{'source': source}),
                    tm)

    def update_entry(self, entry, tm):
        kb_entry = self.kb
        bNew = False
        for param in StaticKnowledge.ID_PARAMS:
            if param in entry:
                if param not in kb_entry:
                    kb_entry[param] = {}
                if entry[param] not in kb_entry[param]:
                    bNew = True
                    kb_entry[param].update({entry[param]: {}})
                kb_entry = kb_entry[param][entry[param]]
        if bNew:
            self.novelty_list += [entry]
        if 'total_count' not in kb_entry:
            kb_entry['total_count'] = 1
            kb_entry['last_seen'] = tm
        elif tm - kb_entry['last_seen'] > 5:
            # not a good way to count stuff, but whatever...
            # relative rarety can be estimated
            kb_entry['total_count'] += 1
            kb_entry['last_seen'] = tm

    def is_known(self, entry):
        kb_entry = self.kb
        flag = False
        for param in StaticKnowledge.ID_PARAMS:
            if param in entry:
                flag = True
                if param not in kb_entry or entry[param] not in kb_entry[param]:
                    return False
                kb_entry = kb_entry[param][entry[param]]
        return flag

def loopOr(fun, arg):
    results = [fun(one_arg) for one_arg in arg]
    return any(results)

class Explore(Switcher):

    def __init__(self, agent):
        super().__init__(agent.rob)
        self.agent = agent
        self.searching = False
        self.block2check = []
        self.pos2check = []
        self.item2pick = []

    def update(self):
        for nov in self.agent.kb.novelty_list:
            if nov['source'] == 'getInventory':
                variant = nov['variant'] + ' ' if 'variant' in nov else ''
                self.rob.sendCommand(['chat', 'I got new item ' + variant + nov['type']])
            elif nov['source'] == 'getNearEntities':
                if 'life' in nov:
                    self.rob.sendCommand(['chat', 'New life detected: ' + nov['name']])
                else: self.item2pick += [nov]
            elif nov['source'] == 'getNearGrid':
                self.block2check += [nov]
            elif nov['source'] == 'getLineOfSights':
                if nov['hitType'] == 'block':
                    self.pos2check += [nov]
                    variant = nov['variant'] + ' ' if 'variant' in nov else ''
                    self.rob.sendCommand(['chat', 'Seeing something new: ' + variant + nov['type']])
        self.agent.kb.novelty_list = []

        # a little bit hacky, but this agent doesn't explore crafting
        if self.delegate is None or not isinstance(self.delegate, Obtain):
            inv = self.rob.cached['getInventory'][0]
            logs = minelogy.get_target_variants({'source': 'getInventory', 'type': 'log'})
            if self.agent.kb.is_known({'source': 'getInventory', 'type': 'cobblestone'}) and \
              not minelogy.isInInventory(inv, {'type': 'stone_pickaxe'}):
               self.delegate = Obtain(self.agent, [{'type': 'stone_pickaxe'}])
            if (loopOr(self.agent.kb.is_known, logs)) and not minelogy.isInInventory(inv, {'type': 'wooden_pickaxe'}):
                self.delegate = Obtain(self.agent, [{'type': 'wooden_pickaxe'}])

        if self.block2check != [] or self.pos2check != [] or self.item2pick != []:
            if self.delegate is None:
                if self.item2pick != []:
                    item = self.item2pick[-1]
                    self.rob.sendCommand(['chat', 'Trying to pick new item: ' + item['name']])
                    self.delegate = ApproachPos(self.agent, [item['x'], item['y'], item['z']])
                    self.item2pick = self.item2pick[:-1]
                elif self.pos2check != []:
                    los = self.pos2check[-1]
                    pos = [los['x'], los['y'], los['z']]
                    self.delegate = FindAndMine(self.agent, [los['type']])
                    #self.delegate = SAnd([ApproachPos(self.agent, pos),
                    #                      LookAndAttackBlock(self.rob, pos)])
                    print("Approaching " + los['type'])
                    self.pos2check = self.pos2check[:-1]
                elif self.block2check != []:
                    while self.block2check != []:
                        block = self.block2check[-1]['type']
                        pos = self.agent.nearestBlock([block])
                        self.block2check = self.block2check[:-1]
                        if pos is not None:
                            print('Sensing something new: ' + block)
                            self.delegate = SAnd([ApproachPos(self.agent, pos),
                                                  LookAt(self.rob, pos)])
                            break
                    # if self.delegate is None -- may happen?
            elif self.searching:
                self.stopDelegate = True
                self.searching = False
            # else do nothing
        elif self.delegate is None:
            self.rob.sendCommand(['chat', 'Searching for something new...'])
            # TODO: non-existing block to search is not the best way; relocate?
            self.delegate = NeuralSearch(self.agent, ['--'])
            self.searching = True
        super().update()


class Explorer(TAgent):

    def __init__(self, miss, visualizer=None, goal=None):
        super().__init__(miss, visualizer)
        self.goal = ListenAndDo(self)
        self.goal.delegate = Explore(self)
        self.kb = StaticKnowledge(self.rob)

    def run(self):
        running = True
        while running:
            sleep(0.05)
            self.rob.observeProcCached()
            self.blockMem.updateBlocks(self.rob)
            self.kb.update()
            self.visualize()
            acts, running = self.goal.cycle()
            for act in acts:
                self.rob.sendCommand(act)
        acts = self.goal.stop()
        for act in acts:
            self.rob.sendCommand(act)


if __name__ == '__main__':
    SCALE = 3
    setup_logger()
    visualizer = Visualizer()
    visualizer.start()
    video_producer = mb.VideoProducer(width=320 * SCALE, height=240 * SCALE, want_depth=False)
    agent_handlers = mb.AgentHandlers(video_producer=video_producer)
    miss = mb.MissionXML(agentSections=[mb.AgentSection(name='Robo',
                agenthandlers=agent_handlers,)])

    world = mb.defaultworld(forceReset="true", seed="151")
    #113 122 127? 128 129+? 130+? 131+?
    miss.setWorld(world)

    agent = Explorer(miss, visualizer=visualizer)
    sleep(4)
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
