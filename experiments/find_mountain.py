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
from behaviours import *
import logging

logger = logging.getLogger(__file__)

SCALE = 2
RESIZE = 1
HEIGHT = 240 * SCALE
WIDTH = 320 * SCALE


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


def show_heatmaps(obs):
    segm_data = obs.getCachedObserve('getNeuralSegmentation')
    heatmaps, img = segm_data
    visualizer('coal_ore', (heatmaps[0, 3].cpu().detach().numpy() * 255).astype(numpy.uint8))


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
    show_img = lambda: visualize(None, get_image(obs.getCachedObserve('getImageFrame'), RESIZE, SCALE))
    show_segm = lambda: visualize(get_image(obs.getCachedObserve('getSegmentationFrame'), RESIZE, SCALE), None)
    neural_callback = NeuralWrapper(obs, RESIZE, SCALE)
                                # cb_name, on_change event, callback
    obs.addCallback('getNeuralSegmentation', 'getImageFrame', neural_callback)

    obs.addCallback(None, 'getImageFrame', show_img)
    obs.addCallback(None, 'getSegmentationFrame', show_segm) 
    # attach visualization callback to getNeuralSegmentation
    obs.addCallback(None, 'getNeuralSegmentation', lambda: show_heatmaps(obs)) 
    mc.safeStart()
    runSkill(obs)
    visualizer.stop()
    sys.exit(0)
