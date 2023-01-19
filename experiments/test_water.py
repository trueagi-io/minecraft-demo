import cv2
from common import *
import numpy
import time
from examples.neural import get_image
from behaviours import TurnTo, PITCH, YAW
import tagilmo.utils.mission_builder as mb
from tagilmo.utils.mission_builder import AgentStart
from tagilmo.utils.vereya_wrapper import MCConnector, RobustObserverWithCallbacks
from tagilmo.utils import segment_mapping
import tagilmo.utils.mission_builder as mb


SCALE = 2 
WIDTH = 320 * SCALE
HEIGHT = 240 * SCALE


def detect_water_see(image):
     blue_lower = numpy.array([111.35, 163.3 , 158.7 ])
     blue_upper = numpy.array([121.84722222, 211.55, 245.])
     see_low = numpy.array([106, 155, 234], dtype=numpy.uint8)
     see_high = numpy.array([126, 175, 254], dtype=numpy.uint8)
 
     hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
     mask = cv2.inRange(hsv, blue_lower, blue_upper)
     mask_see = cv2.inRange(hsv, see_low, see_high)
     return mask, mask_see


def main():
    mc, rob = start_mission()
    mc.safeStart()
    time.sleep(1)
    while True:
        rob.updateAllObservations()
        time.sleep(0.15)
        frame = rob.getCachedObserve('getImageFrame')
        image = get_image(frame, 1, SCALE) 

        mask, mask_see = detect_water_see(image)
        cv2.imshow('img', image)
        cv2.imshow('water', mask)
        cv2.imshow('see', mask_see)
        cv2.waitKey(300)


def start_mission():
    miss = mb.MissionXML()
    colourmap_producer = mb.ColourMapProducer(width=WIDTH, height=HEIGHT)
    video_producer = mb.VideoProducer(width=WIDTH, height=HEIGHT, want_depth=False)
    colourmap_producer = None

    obs = mb.Observations()
    agent_handlers = mb.AgentHandlers(observations=obs)

    agent_handlers = mb.AgentHandlers(observations=obs,
        colourmap_producer=colourmap_producer,
        video_producer=video_producer)

    miss = mb.MissionXML(agentSections=[mb.AgentSection(name='Cristina',
             agenthandlers=agent_handlers)])
    
    # seed 28 - with a see
    world = mb.defaultworld(
        seed='28',
        forceReset="false")

    
    miss.setWorld(world)
    miss.serverSection.initial_conditions.allowedmobs = "Pig Sheep Cow Chicken Ozelot Rabbit Villager"
    # disable passage of time:
    miss.serverSection.initial_conditions.time_pass = 'false'
    miss.serverSection.initial_conditions.time_start = "1000"

    mc = MCConnector(miss)
    obs1 = RobustObserverWithCallbacks(mc)
    return mc, obs1


if __name__ == '__main__':
   main()

