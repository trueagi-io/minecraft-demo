import random
import cv2
import time
import numpy
import logging
import tagilmo.utils.mission_builder as mb
from tagilmo.utils.malmo_wrapper import MalmoConnector
from tagilmo.utils import mapping


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


def read_colormap():
    colormap = dict()
    order = []
    with open('colourmap.txt' , 'rt') as f:
        for line in f:
            r, g, b, name = line.split(',')
            if name.strip() not in order:
                order.append(name.strip())
            colormap[(int(r), int(g), int(b))] = order.index(name.strip()) + 1 
    return colormap, order


def img_to_int1(colormap, order, img):
    img1 = img.copy()
    # exchange first and last channels
    img1[:,:, [0, 2]] = img1[:,:, [2, 0]]
 
    tmp = numpy.zeros(img.shape[:2])
    for k, v in colormap.items():
        num = v 
        start = time.time()
        item_present = ((img1 == k).sum(axis=2) == 3) 
        end = time.time()
        import pdb;pdb.set_trace()
        tmp += item_present * num
        if item_present.max():
            print('found item ', order[v - 1])
    return tmp 

def img_to_int2(_color_codes, order, img):
    img1 = img.copy()
    # exchange first and last channels
    img1[:,:, [0, 2]] = img1[:,:, [2, 0]]
    # Extract color codes and their IDs from input dict
    colors = numpy.array(list(_color_codes.keys()))
    color_ids = numpy.array(list(_color_codes.values()))

    # Initialize output array
    result = numpy.empty((img.shape[0],img.shape[1]),dtype=int)
    result[:] = 0 
    # Finally get the matches and accordingly set result locations
    # to their respective color IDs
    R,C,D = numpy.where((img1 == colors[:,None,None,:]).all(3))
    result[C,D] = color_ids[R]
    return result

def main():
    start = -88, 88
    colormap, order = read_colormap()
    mc = init_mission(None, *start) 
    mc.safeStart()
    while True:
        img = get_img(mc)
        if img is not None:
            if mc.isLineOfSightAvailable(0):
                print(mc.observe[0]['LineOfSight'])

            cv2.imshow('colormap', img)
            cv2.imshow(mapping[120], (img[:,:, 0] == 120) * 255.0)
            cv2.waitKey(1000)
            

main()
