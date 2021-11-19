import cv2
import sys
from common import normAngle, degree2rad
from examples.neural import NeuralWrapper
from b_tree import *
import torch
import os
import math
import numpy
from collections import deque, defaultdict
import time
import tagilmo.utils.mission_builder as mb
from tagilmo.utils.malmo_wrapper import MalmoConnector, RobustObserverWithCallbacks
from tagilmo.utils import segment_mapping
import logging

logger = logging.getLogger(__file__)

SCALE = 2
RESIZE = 1 / SCALE
HEIGHT = 240 * SCALE
WIDTH = 320 * SCALE


class Base:
    def __init__(self, rob):
       self.rob = rob 

    def __call__(self):
        return self.act()


class CheckVisible(Base):
    def __init__(self, rob):
        super().__init__(rob)
        self.found_item = None

    def act(self):
        result = []
        status = 'running'
        segm_data = self.rob.getCachedObserve('getNeuralSegmentation')
        if segm_data is None:
            return status, result
        ray = self.rob.getCachedObserve('getLineOfSights')
        if ray is not None and ray['type'] == 'stone':
            self.found_item = ray 
            return 'success', ['pitch 0', 'turn 0']
        if segm_data is not None:
            heatmaps, img = segm_data
            h, w = heatmaps.shape[-2:]
            size = (h // 10, h // 10)
            stone = heatmaps[0, 4].unsqueeze(0).unsqueeze(0)
            # smooth heatmap a bit
            pooled = torch.nn.functional.avg_pool2d(stone, kernel_size=size, stride=size)
            visualizer('avg_stone', (pooled * 255).long().numpy().astype(numpy.uint8)[0, 0])
            visualizer('stone', (stone * 255).long().numpy().astype(numpy.uint8)[0, 0])
            if pooled.max() < 0.3:
                # can't see stone
                return 'failure', []
            idx = torch.argmax(pooled)
            h_pooled, w_pooled = pooled.shape[-2:]
            h_idx = idx // w_pooled 
            w_idx = idx % w_pooled
            pitch = (h_idx.item() - h_pooled / 2) / 80
            turn = (w_idx.item() - w_pooled / 2) / 80 
            result = ['pitch {0}'.format(pitch), 'turn {0}'.format(turn)]

        return status, result 

YAW = 4
PITCH = 3

class Stabilize(Base):

    def __init__(self, rob):
        super().__init__(rob)
        self.prev_pitch = None

    def act(self):
        result = []
        pitch = 0
        logger.debug('stabilizing')
        pos = self.rob.waitNotNoneObserve('getAgentPos')
        current_pitch = pos[PITCH]
        # proportional control
        # should work good enough, no need for PID
        optimal_pitch = -6
        diff = -(current_pitch - optimal_pitch)
        if abs(diff) < 1.5:
            diff = 0
        pitch = diff / 200 
        result.append('pitch {0}'.format(pitch))
        status = 'running'
        if pitch == 0 and self.prev_pitch == 0:
            status = 'success'
        self.prev_pitch = pitch
        return status, result


class Rotate(Base):
    """rotate 1 rad then return success"""

    def __init__(self, rob):
        self.start_yaw = None
        self.rob = rob

    def act(self):
        result = [] 
        pos = self.rob.waitNotNoneObserve('getAgentPos')
        current_yaw = normAngle(degree2rad(pos[YAW]))
        if self.start_yaw is None:
           self.start_yaw = current_yaw      
        if abs(normAngle(current_yaw - self.start_yaw)) > 1:
            result.append('turn 0')
            return 'success', result
        result.append('turn 0.02')
        return 'running', result


class MountainScan:
    def __init__(self, rob):
        self.rob = rob
        self.root = self._create_tree() 
        self._finished = False
        self.start_yaw = None 

    def _create_tree(self):
        rob = self.rob
        stabilize_rotate = And([Stabilize(rob), Rotate(rob)])
        return Or([CheckVisible(rob), stabilize_rotate])

    def getAngle(self):
        pos = self.rob.waitNotNoneObserve('getAgentPos')
        return normAngle(degree2rad(pos[YAW]))
        
    def act(self):
        assert not self._finished
        if self.start_yaw is None:
            self.start_yaw = self.getAngle()
        status, action = self.root()
        if status ==  'failure': 
             self._finished = True
        if status == 'success':
            # we can either succed by finding a mountain
            # or it just rotate finished
            last_node = self.root.getLastNode()
            if isinstance(last_node, Rotate):
                diff = abs(normAngle(self.start_yaw - self.getAngle()))
                if diff  > 1.7 * math.pi:
                    # we're done
                    self._finished = True
                else:
                    self.root = self._create_tree()
            elif isinstance(last_node, CheckVisible):
                print('found mountain')     
                print(last_node.found_item)
                self._finished = True
            else:
                print('unexpected {0}'.format(last_node))
        return action 

    def finished(self):
        return self._finished

    def precond(self):
        return True


def runSkill(rob):
    skill = MountainScan(rob)
    while True:
        acts = []
        rob.updateAllObservations()
        ray = rob.cached['getLineOfSights']
        if skill.precond() and not skill.finished():
            acts = skill.act()
        if skill.finished():
            return
        logger.info(acts)
        for act in acts:
            rob.sendCommand(act)
        time.sleep(0.2)


def visualize(img, segm):
    if img is not None:
       visualizer('image', img)
    if segm is not None:
       visualizer('segm', segm)

    #visualizer('leaves', (heatmaps[0, 2].cpu().detach().numpy() * 255).astype(numpy.uint8))
    #visualizer('log', (heatmaps[0, 1].cpu().detach().numpy() * 255).astype(numpy.uint8))
    #visualizer('coal_ore', (heatmaps[0, 3].cpu().detach().numpy() * 255).astype(numpy.uint8))


def start_mission():
    miss = mb.MissionXML()
    colourmap_producer = mb.ColourMapProducer(width=WIDTH, height=HEIGHT)
    video_producer = mb.VideoProducer(width=WIDTH, height=HEIGHT, want_depth=False)

    obs = mb.Observations()
    agent_handlers = mb.AgentHandlers(observations=obs)

    agent_handlers = mb.AgentHandlers(observations=obs,
        colourmap_producer=colourmap_producer,
        video_producer=video_producer)

    miss = mb.MissionXML(agentSections=[mb.AgentSection(name='Cristina',
             agenthandlers=agent_handlers,)])

    # good point seed='2', x=-90, y=71, z=375
    # good point seed='3', x=6, y=71, z=350
    world = mb.defaultworld(
        seed='31',
        forceReset="false")

    # good seed with mountains 31
    miss.setWorld(world)
    miss.serverSection.initial_conditions.allowedmobs = "Pig Sheep Cow Chicken Ozelot Rabbit Villager"
    # uncomment to disable passage of time:
    miss.serverSection.initial_conditions.time_pass = 'false'
    miss.serverSection.initial_conditions.time_start = "1000"

    mc = MalmoConnector(miss)
    obs1 = RobustObserverWithCallbacks(mc)
    return mc, obs1


def setup_logger():
    # create logger
    logger = logging.getLogger(__file__)
    logger.propagate = False
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
    from examples.neural import NeuralWrapper, get_image
    from examples.vis import Visualizer
    visualizer = Visualizer() 
    visualizer.start()
    mc, obs = start_mission()
    show_img = lambda: visualize(None, get_image(obs, RESIZE, SCALE, 'getImageFrame'))
    show_segm = lambda: visualize(get_image(obs, RESIZE, SCALE, 'getSegmentationFrame'), None)
    neural_callback = NeuralWrapper(obs, RESIZE, SCALE)
                                # cb_name, on_change event, callback
    obs.addCallback('getNeuralSegmentation', 'getImageFrame', neural_callback)

    obs.addCallback(None, 'getImageFrame', show_img)
    obs.addCallback(None, 'getSegmentationFrame', show_segm) 
    mc.safeStart()
    runSkill(obs)
    visualizer.stop()
    sys.exit(0)
