import unittest
import logging
from tagilmo import VereyaPython
import json
import time
import os
import warnings
import tagilmo.utils.mission_builder as mb
from tagilmo.utils.vereya_wrapper import MCConnector, RobustObserver
from base_test import BaseTest


def init_mission(mc, start_x, start_z, seed, forceReset="false",
                 forceReuse="false", start_y=78, worldType = "default", drawing_decorator=None, serverIp=None, serverPort=0):


    want_depth = False
    video_producer = mb.VideoProducer(width=320 * 4,
                                      height=240 * 4, want_depth=want_depth)

    obs = mb.Observations()
    obs.gridNear = [[-2, 2], [-2, 2], [-2, 2]]


    agent_handlers = mb.AgentHandlers(observations=obs, video_producer=video_producer)

    print('starting at ({0}, {1})'.format(start_x, start_y))

    start = [start_x, start_y, start_z, 1]
    if all(x is None for x in [start_x, start_y, start_z]):
        start = None
    #miss = mb.MissionXML(namespace="ProjectMalmo.microsoft.com",
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

def count_items(inv, name):
    result = 0
    for elem in inv:
        if elem['type'] == name:
            result += elem['quantity']
    return result
