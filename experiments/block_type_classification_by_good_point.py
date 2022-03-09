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

from examples.agent import TAgent
import logging
from examples.vis import Visualizer

import imageio

import sys

# git clone https://github.com/singnet/image-matching
# path_to_image_matching is the path to directory with image-matching api
# you also need actual snapshot, e. g. https://github.com/singnet/image-matching/blob/master/fem/snapshots/mine8.pt
path_to_image_matching = "/home/oleg/Work/image-matching"
sys.path.append(os.path.abspath(path_to_image_matching))

from fem.goodpoint import GoodPoint


SCALE = 4


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


class BlockGPDescAnalyzer:

    def __init__(self, rob):
        # dict of pairs (current avg, num of observations)
        self.avg_block_dscr_hist = {}
        self.rob = rob

    def _getLocalDscr(self):
        wnd_thr = 5
        img = get_image(self.rob.getCachedObserve('getImageFrame'), SCALE, SCALE)
        # print(torch.cuda.is_available())
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # print('using device {0}'.format(device))
        # here you need actual snapshot
        weights = 'mine8.pt'
        gp = GoodPoint(n_channels=3,
                       activation=torch.nn.LeakyReLU(),
                       grid_size=8,
                       batchnorm=False,
                       dustbin=0,
                       desc_out=8).to(device)

        if os.path.exists(weights):
            state_dict = torch.load(weights, map_location=device)
            # print("loading weights from {0}".format(weights))
            gp.load_state_dict(state_dict['superpoint'])

        wnd_thr = 30
        height, width, channels = img.shape
        y = height // 2
        x = width // 2
        points = numpy.asarray([[y, x]])
        crop_img = img[y - wnd_thr:y + wnd_thr, x - wnd_thr:x + wnd_thr, 0:3]
        descriptors = gp.get_descriptors(crop_img, points)
        return descriptors.cpu().detach().numpy()

    def collectStat(self, use_exp_avg=False, alpha=0.5):
        # alpha (0, 1) only used when use_exp_avg is True
        los = self.rob.cached['getLineOfSights'][0]
        loc_dscr = self._getLocalDscr()
        if los is None and not(self.avg_block_dscr_hist.get('sky') is None):
            curr_hue = self.avg_block_dscr_hist['sky'][0]
            curr_num = self.avg_block_dscr_hist['sky'][1] + 1
            if use_exp_avg:
                self.avg_block_dscr_hist['sky'][0] = curr_hue + (loc_dscr - curr_hue) * alpha
            else:
                self.avg_block_dscr_hist['sky'][0] = curr_hue + (loc_dscr - curr_hue) / curr_num
            self.avg_block_dscr_hist['sky'][1] = curr_num
            return False
        elif los is None:
            self.avg_block_dscr_hist['sky'] = [loc_dscr, 1]
            return True
        if not(self.avg_block_dscr_hist.get(los['type']) is None):
            curr_hue = self.avg_block_dscr_hist[los['type']][0]
            curr_num = self.avg_block_dscr_hist[los['type']][1] + 1
            if use_exp_avg:
                self.avg_block_dscr_hist[los['type']][0] = curr_hue + (loc_dscr - curr_hue) * alpha
            else:
                self.avg_block_dscr_hist[los['type']][0] = curr_hue + (loc_dscr - curr_hue) / curr_num
            self.avg_block_dscr_hist[los['type']][1] = curr_num
            return False
        else:
            self.avg_block_dscr_hist[los['type']] = [loc_dscr, 1]
            return True

    def saveStatJson(self, path_plus_name):
        with open(path_plus_name, "w") as fp:
            json.dump(self.avg_block_dscr_hist, fp)
            fp.close()

    def searchNNBlock(self):
        loc_hue = self._getLocalDscr()
        vals = list(self.avg_block_dscr_hist.values())
        arr = numpy.array([val[0] for val in vals])
        idx = numpy.argmin(numpy.linalg.norm(arr - loc_hue))
        return list(self.avg_block_dscr_hist.keys())[idx]


class MockAgent(TAgent):

    def loop(self):
        block_hue_analyzer = BlockGPDescAnalyzer(self.rob)
        while True:
            sleep(0.05)
            self.rob.updateAllObservations()
            self.visualize()
            block_hue_analyzer.collectStat(use_exp_avg=True, alpha=0.1)
            if self.rob.cached['getLineOfSights'][0] is None:
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


if __name__ == '__main__':
    setup_logger()
    visualizer = Visualizer()
    visualizer.start()
    video_producer = mb.VideoProducer(width=320 * SCALE, height=240 * SCALE, want_depth=False)
    agent_handlers = mb.AgentHandlers(video_producer=video_producer)
    miss = mb.MissionXML(agentSections=[mb.AgentSection(name='Cristina',
             agenthandlers=agent_handlers,)])

    world = mb.flatworld("3;7,25*1,3*3,2;1;stronghold,biome_1,village,decoration,dungeon,lake,mineshaft,lava_lake",
                         seed='43', forceReset="false")
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
