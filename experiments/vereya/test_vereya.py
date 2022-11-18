import cv2
import torch
import os
import math
import numpy
from collections import deque, defaultdict
import time
import tagilmo.utils.mission_builder as mb
from tagilmo.utils.malmo_wrapper import MalmoConnector, RobustObserver
from tagilmo.utils import segment_mapping


IMAGE = 0
SEGMENT = 1
BLOCKS = 2
WIDTH = 320
HEIGHT = 240


def start_mission():
    miss = mb.MissionXML()
    colourmap_producer = mb.ColourMapProducer(width=WIDTH, height=HEIGHT)
    video_producer = mb.VideoProducer(width=WIDTH, height=HEIGHT, want_depth=False)

    obs = mb.Observations(bNearby=True, bRecipes=True)
    agent_handlers = mb.AgentHandlers(observations=obs)

    agent_handlers = mb.AgentHandlers(observations=obs,
        colourmap_producer=colourmap_producer,
        video_producer=video_producer)

    miss = mb.MissionXML(agentSections=[mb.AgentSection(name='Cristina',
             agenthandlers=agent_handlers,)])
             # agenthandlers=agent_handlers,)], namespace='ProjectMalmo.microsoft.com')

    # good point seed='2', x=-90, y=71, z=375
    # good point seed='3', x=6, y=71, z=350
    # good point seed='31'
    world = mb.defaultworld(
        seed='5',
        forceReset="false",
        forceReuse="true")

    world1 = mb.flatworld("3;7,25*1,3*3,2;1;stronghold,biome_1,village,decoration,dungeon,lake,mineshaft,lava_lake",
            seed='43',
            forceReset="false")
    miss.setWorld(world)
    miss.serverSection.initial_conditions.allowedmobs = "Pig Sheep Cow Chicken Ozelot Rabbit Villager"
    # uncomment to disable passage of time:
    miss.serverSection.initial_conditions.time_pass = 'false'
    miss.serverSection.initial_conditions.time_start = "1000"
    mc = MalmoConnector(miss)
    obs1 = RobustObserver(mc)
    return mc, obs1



def get_img(mc):
    img_frame = mc.waitNotNoneObserve('getImageFrame')
    if img_frame is None:
        return None, None
    pos = numpy.asarray((img_frame.pitch, img_frame.yaw, img_frame.xPos, img_frame.yPos, img_frame.zPos))
    return pos, img_frame


def get_segment(mc):
    img_frame = mc.waitNotNoneObserve('getSegmentationFrame')
    if img_frame is None:
        return None, None
    pos = numpy.asarray((img_frame.pitch, img_frame.yaw, img_frame.xPos, img_frame.yPos, img_frame.zPos))
    return pos, img_frame


def get_distance(mc):
    aPos = mc.getAgentPos()
    coords1 = aPos[:3]
    coords = [mc.getLineOfSight('x'),
              mc.getLineOfSight('y'),
              mc.getLineOfSight('z')]

    height = 1.6025
    coords1[1] += height

    dist = numpy.linalg.norm(numpy.asarray(coords) - numpy.asarray(coords1), 2)
    return dist


def extract(data):
    img_data = numpy.frombuffer(data, dtype=numpy.uint8)
    img_data = img_data.reshape((HEIGHT, WIDTH, 3))
    return img_data


def main():
    from tagilmo import VereyaPython

    mc, obs = start_mission()
    mc.safeStart()
    mc.sendCommand('recipes')
    while True:
        cv2.waitKey(300)
        obs.clear()
        pos1, img = get_img(obs)
        if img is not None:
            img = extract(img.pixels)
            # print(mc.getNearEntities())
            if mc.observe[0] is not None and 'recipes' in mc.observe[0]:
                print("got recipes")
                mc.sendCommand('recipes off')
            cv2.imshow('img', img)
main()
