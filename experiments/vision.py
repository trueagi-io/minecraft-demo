import logging
import pickle
import math
import random
import time
import torch
from torch import nn
from torch.nn import functional as F
import os
import network
import numpy

import tagilmo.utils.mission_builder as mb
from tagilmo.utils.malmo_wrapper import MalmoConnector
import common
from common import stop_motion, grid_to_vec_walking, \
    direction_to_target, normAngle

import tree
from goodpoint import GoodPoint


def load_agent(path):
    gp = GoodPoint(8, 18, n_channels=3, batchnorm=False)
    if os.path.exists(path):
        location = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info('loading model from agent_tree.pth')
        data = torch.load(path, map_location=location)
        gp.load_state_dict(data, strict=False)
    return gp


class Trainer(tree.Trainer):
    want_depth = True

    def __init__(self, agent, mc, optimizer, eps, train=True):
        super().__init__(agent, mc, optimizer, eps, train)

    def _random_turn(self):
        turn = numpy.random.random() * random.choice([-1, 1])
        pitch = numpy.random.random() * random.choice([-0.2, 0.0])
        self.act(["turn {0}".format(turn)])
        self.act(["pitch {0}".format(pitch)])
        time.sleep(0.5)
        stop_motion(self.mc)

    def observe_by_line(self):
        visible = self.mc.getLineOfSight('type')
        result = [visible,
                  self.mc.getLineOfSight('x'),
                  self.mc.getLineOfSight('y'),
                  self.mc.getLineOfSight('z')]
        return result

    # Look at a specified location
    def lookAt(self, pitch_new, yaw_new):
        mc = self.mc
        print('look at')
        for t in range(2000):
            time.sleep(0.02)
            mc.observeProc()
            aPos = mc.getAgentPos()
            if aPos is None:
                continue
            current_pitch = normAngle(aPos[3]*math.pi/180.)
            current_yaw = normAngle(aPos[4]*math.pi/180.)
            pitch = normAngle(normAngle(pitch_new) - current_pitch)
            yaw = normAngle(normAngle(yaw_new) - current_yaw)
            if abs(pitch)<0.02 and abs(yaw)<0.02: break
            yaw = yaw * 0.5
            while abs(yaw) > 1:
                yaw *= 0.8
            pitch = pitch * 0.5
            while abs(pitch) > 1:
                pitch *= 0.8
            mc.sendCommand("turn " + str(yaw))
            mc.sendCommand("pitch " + str(pitch))
        mc.sendCommand("turn 0")
        mc.sendCommand("pitch 0")

    def run_episode(self):
        """
        Collect data for visual system

        1) apply random rotation
        2) collect 3d points in regular grid covering an image
        3) map point to 2d images
        """
        mc = self.mc
        import pdb;pdb.set_trace()
        #self._random_turn()
        K = torch.as_tensor([ 1.6740810033016248e+02, 0., 160., 0., 1.6740810033016248e+02,
       120., 0., 0., 1. ]).reshape((3,3))
        K_inv = torch.inverse(K).numpy()
        time.sleep(1)
        state = self.collect_state()
        image = state['image']
        _, height, width = image.shape
        center_x = round(width / 2)
        center_y = round(height / 2)
        episode_data = dict()
        episode_data['origin'] = state
        pitch, yaw = state['pitch_yaw']
        # unit vector
        unit_pixel = [center_x, center_y, 1]
        # go from top left to bottom right
        X_pixel = [1, 1, 1]
        # x, y, depth
        # from pixels to centimeters
        X_cm = K_inv @ X_pixel
        unit_cm = K_inv @ unit_pixel
        # yaw
        vec_yaw = X_cm.copy()
        vec_yaw[1] = unit_cm[1]
        angle_yaw = common.vectors_angle(vec_yaw, unit_cm)

        # pitch
        vec_pitch = X_cm.copy()
        vec_pitch[0] = unit_cm[0]
        angle_pitch = common.vectors_angle(vec_pitch, unit_cm)

        angleMaxW = abs(angle_yaw)
        angleMaxH = abs(angle_pitch)
        w_steps = 35
        h_steps = 20
        w_range = numpy.arange(yaw, yaw + angleMaxW * 2, angleMaxW * 2 / w_steps) - angleMaxW
        h_range = numpy.arange(pitch, pitch + angleMaxH * 2, angleMaxH * 2 / h_steps) - angleMaxH
        for i in range(w_steps):
            for j in range(h_steps):
                yaw_new = normAngle(w_range[i])
                pitch_new = normAngle(h_range[j])
                self.lookAt(pitch_new, yaw_new)
                time.sleep(0.1)
                state = self.collect_state()
                episode_data[(i, j)] = state
        pickle.dump(episode_data, open('episode7.pkl', 'wb'))
        return episode_data

    def collect_state(self):
        while True:
            self.mc.observeProc()
            data = self.mc.getImage()
            aPos = self.mc.getAgentPos()

            if not any(x is None for x in (data, aPos)):
                pitch_raw = aPos[3] * math.pi/180.
                yaw_raw = aPos[4] * math.pi/180.
                self_pitch = normAngle(aPos[3] * math.pi/180.)
                self_yaw = normAngle(aPos[4] * math.pi/180.)

                data = data.reshape((240, 320, 3 + self.want_depth)).transpose(2, 0, 1) / 255.
                pitch_yaw = torch.as_tensor([self_pitch, self_yaw])
                pitch_yaw_raw = torch.as_tensor([pitch_raw, yaw_raw])
                height = 1.6025
                x, y, z = aPos[0:3]
                y += height
                visible = self.observe_by_line()
                return dict(image=torch.as_tensor(data).float(),
                            pitch_yaw=pitch_yaw,
                            pitch_yaw_raw=pitch_yaw_raw,
                            coordinates=[x, y, z],
                            visible=visible)
            else:
                time.sleep(0.05)

    @classmethod
    def init_mission(cls, i, mc):
        miss = mb.MissionXML()
        video_producer = mb.VideoProducer(width=320, height=240, want_depth=cls.want_depth)

        obs = mb.Observations()
        agent_handlers = mb.AgentHandlers(observations=obs,
            all_str=tree.mission_ending, video_producer=video_producer)
        # a tree is at -18, 15
        start_x = -17
        start_y = 13

        logging.info('starting at ({0}, {1})'.format(start_x, start_y))
        miss = mb.MissionXML(agentSections=[mb.AgentSection(name='Cristina',
                 agenthandlers=agent_handlers,
                                          #    depth
                 agentstart=mb.AgentStart([start_x, 30.0, start_y, 1]))])

        miss.setWorld(mb.flatworld("3;7,25*1,3*3,2;1;stronghold,biome_1,village,decoration,dungeon,lake,mineshaft,lava_lake",
            seed='43',
            forceReset="false"))
        miss.serverSection.initial_conditions.allowedmobs = "Pig Sheep Cow Chicken Ozelot Rabbit Villager"

        if mc is None:
            mc = MalmoConnector(miss)
        else:
            mc.setMissionXML(miss)
        return mc

