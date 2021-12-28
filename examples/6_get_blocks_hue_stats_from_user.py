import math
import torch
import random
import threading
import cv2
import numpy
import os
import json
from time import sleep, time
import tagilmo.utils.mission_builder as mb
from examples import minelogy
from examples.agent import TAgent
import logging
from vis import Visualizer

from tagilmo.utils.mathutils import *


def process_pixel_data(pixels, resize, scale):
    img_data = numpy.frombuffer(pixels, dtype=numpy.uint8)
    img_data = img_data.reshape((240 * scale, 320 * scale, 3))
    if resize != 1:
        height, width, _ = img_data.shape
        img_data = cv2.resize(img_data, (int(width * resize), int(height * resize)),
            fx=resize, fy=resize, interpolation=cv2.INTER_NEAREST)

    return img_data


def get_image(img_frame, resize, scale):
    if img_frame is not None:
        return process_pixel_data(img_frame.pixels, resize, scale)
    return None


class BlockHueAnalyzer:

    def __init__(self, rob):
        # dict of pairs (current avg, num of observations)
        self.avg_block_hue_hist = {}
        self.rob = rob

    def _getLocalHue(self):
        wnd_thr = 5
        img = get_image(self.rob.getCachedObserve('getImageFrame'), 4, 4)
        bgr = img[:, :, 0:3]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        height, width, channels = img.shape
        y = height//2
        x = width//2
        crop_img = h[y-wnd_thr:y+wnd_thr, x-wnd_thr:x+wnd_thr]
        avg = numpy.mean(crop_img)
        return avg

    def collectStat(self):
        los = self.rob.cached['getLineOfSights'][0]
        loc_hue = self._getLocalHue()
        if los is None and not(self.avg_block_hue_hist.get('sky') is None):
            curr_hue = self.avg_block_hue_hist['sky'][0]
            curr_num = self.avg_block_hue_hist['sky'][1] + 1
            self.avg_block_hue_hist['sky'][0] = curr_hue + (loc_hue - curr_hue) / curr_num
            self.avg_block_hue_hist['sky'][1] = curr_num
            return False
        elif los is None:
            self.avg_block_hue_hist['sky'] = [loc_hue, 1]
            return True
        if not(self.avg_block_hue_hist.get(los['type']) is None):
            curr_hue = self.avg_block_hue_hist[los['type']][0]
            curr_num = self.avg_block_hue_hist[los['type']][1] + 1
            self.avg_block_hue_hist[los['type']][0] = curr_hue + (loc_hue - curr_hue)/curr_num
            self.avg_block_hue_hist[los['type']][1] = curr_num
            return False
        else:
            self.avg_block_hue_hist[los['type']] = [loc_hue, 1]
            return True

    def saveStatJson(self, path_plus_name):
        with open(path_plus_name, "w") as fp:
            json.dump(self.avg_block_hue_hist, fp)
            fp.close()

    def searchNNBlock(self):
        loc_hue = self._getLocalHue()
        vals = list(self.avg_block_hue_hist.values())
        arr = numpy.array([val[0] for val in vals])
        idx = numpy.argmin(numpy.abs(arr - loc_hue))
        return list(self.avg_block_hue_hist.keys())[idx]


class MockAgent(TAgent):

    def loop(self):
        block_hue_analyzer = BlockHueAnalyzer(self.rob)
        while True:
            sleep(0.05)
            self.rob.updateAllObservations()
            self.visualize()
            block_hue_analyzer.collectStat()
            if (self.rob.cached['getLineOfSights'][0] is None):
                print('Expected object: sky.  Seen by agent as {}'.format(block_hue_analyzer.searchNNBlock()))
            else:
                print('Expected object: {}.  Seen by agent as {}'.format(self.rob.cached['getLineOfSights'][0]['type'],
                                                                         block_hue_analyzer.searchNNBlock()))
        return True

def setup_logger():
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s: %(levelname)-8s %(message)s')
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(logging.DEBUG)
    # add ch to logger
    logger.addHandler(ch)

SCALE = 4

if __name__ == '__main__':
    setup_logger()
    visualizer = Visualizer()
    visualizer.start()
    video_producer = mb.VideoProducer(width=320 * SCALE, height=240 * SCALE, want_depth=False)
    agent_handlers = mb.AgentHandlers(video_producer=video_producer)
    miss = mb.MissionXML(agentSections=[mb.AgentSection(name='Cristina',
             agenthandlers=agent_handlers,)])


    world = mb.flatworld("3;7,25*1,3*3,2;1;stronghold,biome_1,village,decoration,dungeon,lake,mineshaft,lava_lake", seed='43', forceReset="false")
    miss.serverSection.initial_conditions.time_pass = 'false'
    miss.serverSection.initial_conditions.time_start = "1000"
    world1 = mb.defaultworld(forceReset="true")
    miss.setWorld(world1)
    # miss.serverSection.initial_conditions.allowedmobs = "Pig Sheep Cow Chicken Ozelot Rabbit Villager"
    agent = MockAgent(miss, visualizer=visualizer)
    agent.rob.sendCommand("chat /difficulty peaceful")
    # agent.loop()
    agent.loop()

    visualizer.stop()
