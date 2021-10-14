import random
import cv2
import time
import numpy
import logging
import tagilmo.utils.mission_builder as mb
from tagilmo.utils.malmo_wrapper import MalmoConnector
from tagilmo.utils import segment_mapping


mission_ending = """
<MissionQuitCommands quitDescription="give_up"/>
<RewardForMissionEnd>
  <Reward description="give_up" reward="243"/>
</RewardForMissionEnd>
"""


def init_mission(mc, start_x=None, start_y=None):
    miss = mb.MissionXML()
    colourmap_producer = mb.ColourMapProducer(width=320 * 4, height=240 * 4)
    video_producer = mb.VideoProducer(width=320 * 4, height=240 * 4, want_depth=False)    
    video_producer = None # use None; there are races in client library

    obs = mb.Observations()

    obs = mb.Observations()
    obs.gridNear = [[-1, 1], [-2, 1], [-1, 1]]


    agent_handlers = mb.AgentHandlers(observations=obs,
        all_str=mission_ending)

    agent_handlers = mb.AgentHandlers(observations=obs,
        all_str=mission_ending, colourmap_producer=colourmap_producer,
        video_producer=video_producer)
    # a tree is at -18, 15
    if start_x is None:
        center_x = -18
        center_y = 15

        start_x = center_x + random.choice(numpy.arange(-329, 329))
        start_y = center_y + random.choice(numpy.arange(-329, 329))

    logging.info('starting at ({0}, {1})'.format(start_x, start_y))

    miss = mb.MissionXML(agentSections=[mb.AgentSection(name='Cristina',
             agenthandlers=agent_handlers,#)])
                                      #    depth
             agentstart=mb.AgentStart([start_x, 30.0, start_y, 1]))])
    world = mb.flatworld("3;7,25*1,3*3,2;1;stronghold,biome_1,village,decoration,dungeon,lake,mineshaft,lava_lake", 
        seed='43',
        forceReset="false")
    miss.setWorld(world)
    miss.serverSection.initial_conditions.allowedmobs = "Pig Sheep Cow Chicken Ozelot Rabbit Villager"
    # uncomment to disable passage of time:
    miss.serverSection.initial_conditions.time_pass = 'false'
    miss.serverSection.initial_conditions.time_start = "1000"

    if mc is None:
        mc = MalmoConnector(miss)
    else:
        mc.setMissionXML(miss)
    return mc


def get_img(mc):
    mc.observeProc()
    time.sleep(0.2)
    img_data = mc.getImage()
    if img_data is not None:
        img_data = img_data.reshape((240 * 4, 320 * 4, 3))
        return img_data


def main():
    start = -88, 88
    mc = init_mission(None, *start) 
    mc.safeStart()
    while True:
        img = get_img(mc)
        if img is not None:
            if mc.isLineOfSightAvailable(0):
                print(mc.observe[0]['LineOfSight'])

            cv2.imshow('colormap', img)
            cv2.imshow(segment_mapping[120], (img[:,:, 0] == 120) * 255.0)
            cv2.waitKey(1000)
            

main()
