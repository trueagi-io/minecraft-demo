import random
import time
import numpy
import logging
import tagilmo.utils.mission_builder as mb
from tagilmo.utils.malmo_wrapper import MalmoConnector, RobustObserver



mission_ending = """
<MissionQuitCommands quitDescription="give_up"/>
<RewardForMissionEnd>
  <Reward description="give_up" reward="243"/>
</RewardForMissionEnd>
"""


def init_mission(mc, start_x=None, start_y=None, use_video=False, use_colormap=False):
    miss = mb.MissionXML()
    colourmap_producer = None
    if use_colormap:
        colourmap_producer = mb.ColourMapProducer(width=320 * 4, height=240 * 4)
    video_producer = None
    if use_video:
        video_producer = mb.VideoProducer(width=320 * 4, height=240 * 4, want_depth=False)    

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
    obs1 = RobustObserver(mc)
    return mc, obs1


def get_img(mc):
    img_data = mc.waitNotNoneObserve('getImageFrame')
    if img_data is not None:
        img_data = img_data.reshape((240 * 4, 320 * 4, 3))
        return img_data

def get_segment(mc):
    img_data = mc.waitNotNoneObserve('getSegmentationFrame')
    if img_data is not None:
        img_data = img_data.reshape((240 * 4, 320 * 4, 3))
        return img_data


def main_video_img():
    print('testing original')
    start = -88, 88
    mc, obs = init_mission(None, *start, use_video=True, use_colormap=False)
    mc.safeStart()
    time.sleep(5)
    start = time.time()
    count = 0
    while count < 10:
        obs.clear()
        img = get_img(obs)
        count += 1
    end = time.time()
    print('10 frames in ', end - start)

          
def main_segm_video():
    print('testing segmentation + original')
    start = -88, 88
    mc, obs = init_mission(None, *start, use_video=True, use_colormap=True)
    mc.safeStart()
    time.sleep(5)
    start = time.time()
    count = 0
    while count < 10:
        obs.clear()
        segm = get_segment(obs)
        img = get_img(obs)
        count += 1
    end = time.time()
    print('10 frames in ', end - start)


def main_segm():
    print('testing segmentation')
    start = -88, 88
    mc, obs = init_mission(None, *start, use_video=False, use_colormap=True)
    mc.safeStart()
    time.sleep(5)
    start = time.time()
    count = 0
    while count < 10:
        obs.clear()
        segm = get_segment(obs)
        count += 1
    end = time.time()
    print('10 frames in ', end - start)
             

main_video_img()
main_segm_video()
main_segm()
