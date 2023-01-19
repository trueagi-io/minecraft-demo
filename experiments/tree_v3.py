"""
Environment and agent for the task of turning towards a tree
"""
import logging
import math
import random
import time
import torch
from torch import nn
from torch.nn import functional as F
import os
import network
import numpy
import cv2
from collections import deque, defaultdict

import tagilmo.utils.mission_builder as mb
from tagilmo.utils.vereya_wrapper import MCConnector
import common
from common import stop_motion, grid_to_vec_walking, direction_to_target
from tagilmo.utils.mathutils import toRadAndNorm

from network import QVisualNetwork



mission_ending = """
<MissionQuitCommands quitDescription="give_up"/>
<RewardForMissionEnd>
  <Reward description="give_up" reward="243"/>
</RewardForMissionEnd>
"""


class DeadException(RuntimeError):
    def __init__(self):
        super().__init__("it's dead")


class QVisualNetworkTree(QVisualNetwork):
    def __init__(self, n_prev_images, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_prev_images = n_prev_images
        num = kwargs.get('num', 128)
        num1 = num * 2
        # fully connected
        self.q_value = nn.Sequential(
            nn.Linear(num + (28 * 8 * 8 + 28 * 4 * 4 + 28) * self.n_prev_images, num1),
            self.activation,
            nn.Linear(num1, num1),
            self.activation,
            nn.Linear(num1, num1),
            self.activation,
            nn.Linear(num1, self.n_actions))
        self.residual = False
        self.state_queue = deque(maxlen=2)

    def forward(self, data):
        """

        Parameters:
        data: dict
            expected keys 'image' - tensor of images (batch, T, channels, height, width)
            'state' - vector
        """
        x = data['images'].to(next(self.conv1a.parameters()))
        if len(x.shape) == 4:
            x = x.unsqueeze(0)

        # images are stacked along channel dimention,
        # they need to be translated to batch dimention
        B, T, C, H, W = x.shape
        x = self.vgg(x.view(-1, C, H, W))
        visual_data = self.pooling(x).view(B, T, -1)

        state = data['state']
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        state_data = state.to(next(self.conv1a.parameters()))

        # yaws + dists + heights + actions
        angle_emb = common.angle_embed(state_data[:, 2:])
        actions = state_data[:, :2]
        state_data = torch.cat([angle_emb, actions], dim=1)
        state_emb = self.pos_emb(state_data)
        visual_pos_emb = torch.cat([visual_data.view(B, -1), state_emb], dim=1)
        result = self.q_value(visual_pos_emb)
        if torch.isnan(result).any().item():
            import pdb;pdb.set_trace()

        return result


def load_agent(path):
    # possible actions are
    # move[-1, 1],
    # strafe[-1, 1]
    # pitch[-1, 1]
    # turn[-1, 1]
    # jump 0/1

    # discreet actions
    # "move -0.5" "jump_forward",
    action_names = ["turn 0.15", "turn -0.15", "turn 0.01",
                    "turn 0.01", 'pitch 0.1', 'pitch -0.1',
                    'pitch 0.01', 'pitch -0.01']
    actionSet = [network.CategoricalAction(action_names)]

    transformer = common.make_noisy_transformers()
    policy_net = QVisualNetworkTree(1, actionSet, 0, 34,  n_channels=3, activation=nn.LeakyReLU(), batchnorm=False, num=256)
    target_net = QVisualNetworkTree(1, actionSet, 0, 34,  n_channels=3, activation=nn.LeakyReLU(), batchnorm=False, num=256)

    batch_size = 20
    my_simple_agent = network.DQN(policy_net, target_net, 0.9,
                                  batch_size, 450, capacity=2000,
                                  transform=transformer)

    if os.path.exists('agent_tree.pth'):
        location = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info('loading model from agent_tree.pth')
        data = torch.load('agent_tree.pth', map_location=location)
        my_simple_agent.load_state_dict(data, strict=False)

    return my_simple_agent


class Trainer(common.Trainer):
    want_depth = False

    def __init__(self, agent, mc, optimizer, eps, train=True):
        super().__init__(train)
        self.agent = agent
        self.mc = mc
        self.optimizer = optimizer
        self.eps = eps
        self.agent.to(self.device)
        self.state_queue = deque(maxlen=2)

    def is_tree_visible(self):
        logging.debug(self.mc.getLineOfSight('type'))
        if self.mc.getLineOfSight('type') in ['log', 'leaves']:
            return [self.mc.getLineOfSight('type'),
                    self.mc.getLineOfSight('x'),
                    self.mc.getLineOfSight('y'),
                    self.mc.getLineOfSight('z')]
        return None

    def collect_visible(self, data, coords1):
        visible = self.mc.getLineOfSight('type')
        if visible is not None:
            coords = [self.mc.getLineOfSight('x'),
                      self.mc.getLineOfSight('y'),
                      self.mc.getLineOfSight('z')]
            height = 1.6025
            coords1[1] += height
            dist = numpy.linalg.norm(numpy.asarray(coords) - numpy.asarray(coords1), 2)
            data['visible'] = [visible] + coords + [dist]


    def collect_state(self):
        """
        state has format:
            image, prev_action, prev_prev_action,
            pitch, prev_pitch,
            yaw - prev_yaw, prev_yaw - prev_prev_yaw
        """
        mc = self.mc
        mc.observeProc()
        aPos = mc.getAgentPos()
        img_data = self.mc.getImage()
        logging.debug(aPos)
        while aPos is None or (img_data is None):
            time.sleep(0.05)
            mc.observeProc()
            aPos = mc.getAgentPos()
            img_data = self.mc.getImage()
        height = aPos[1]
        if height < 30: # avoid ponds and holes
            raise DeadException()
        pitch = aPos[3]
        yaw = aPos[4]
        data = dict()
        img_data = img_data.reshape((240 * 4, 320 * 4, 3 + self.want_depth))
        img_data = cv2.resize(img_data, (320, 240))
        img = img_data.reshape((240, 320, 3 + self.want_depth)).transpose(2, 0, 1) / 255.

        img = torch.as_tensor(img).float()
        data['image'] = img
        actions = []

        imgs = [torch.as_tensor(img)]
        yaws = [yaw]
        pitches = [pitch]
        for item in reversed(self.state_queue):
            actions.append(item['action'])
            yaws.append(item['yaw'])
            pitches.append(item['pitch'])
        while len(yaws) < 3:
            actions.append(torch.as_tensor(-1).to(img))
            yaws.append(torch.as_tensor(yaw).to(img))
            pitches.append(torch.as_tensor(pitch).to(img))

        # use relative change for yaws
        for i in range(len(yaws) - 1):
            yaws[i] =  toRadAndNorm(yaws[i] - yaws[i + 1])
        yaws.pop()
        pitches.pop()
        state = torch.as_tensor(actions + pitches + yaws)
        data.update(dict(state=state,
                         images=torch.stack(imgs),
                         image=img,
                         reward=torch.as_tensor(0),
                         yaw=torch.as_tensor(yaw),
                         pitch=torch.as_tensor(pitch)
                         ))
        return data

    def _random_turn(self):
        turn = numpy.random.random() * random.choice([-1, 1])
        pitch = numpy.random.random() * random.choice([-1, 1])
        self.act(["turn {0}".format(turn)])
        self.act(["pitch {0}".format(pitch)])
        time.sleep(0.5)
        stop_motion(self.mc)

    def run_episode(self):
        """ Deep Q-Learning episode
        """
        self.agent.clear_state()
        mc = self.mc
        # apply random turn and pitch
        self._random_turn()
        logging.debug('memory: %i', self.agent.memory.position)
        self.agent.train()

        max_t = 80
        eps_start = self.eps
        eps_end = 0.05
        eps_decay = 0.99

        eps = eps_start

        total_reward = 0

        t = 0

        # pitch, yaw, xpos, ypos, zpos
        prev_pos = None
        prev_target_dist = None
        prev_life = 20
        solved = False

        while True:
            t += 1
            reward = -0.2
            try:
                data = self.collect_state()
                target = self.is_tree_visible()
            except DeadException:
                stop_motion(mc)
                # should not die is this mission
                # so don't add this event to the replay buffer
                reward = 0
                logging.warning("died at step %i", t)
                break
            if self.state_queue:
                # use only dist change for now
                life = mc.getLife()
                logging.debug('current life %f', life)
                if life == 0:
                    # should not die is this mission
                    break
                solved, reward = self.is_solved(target)
                if solved:
                    logging.info('solved in %i steps', t)
                    self.agent.push_final(reward)
                    self.mc.sendCommand("quit")
                    break
                else:
                    r = reward
                    if reward > 0:
                        for item in self.state_queue:
                            if 'reward' in item and item['reward'] > 0:
                                r += item['reward']
                        if r >= 3:
                            solved = True
                            logging.info('solved in %i steps', t)
                            reward = 45
                            self.agent.push_final(reward)
                            self.mc.sendCommand("quit")
                            break
                data['reward'] = torch.as_tensor(reward)
                #if 'stop' in new_actions:
                #    # either it solved, or the tree is blocked
                #    if not solved:
                #        mc.sendCommand('move 1')
                #        time.sleep(5)
                #        stop_motion(mc)
                #        self.collect_state()
                #        target = self.is_tree_visible()
                #        solved, reward = self.is_solved(target)
                #    if solved:
                #        logging.debug('solved in %i steps', t)
                #        reward += 5
                #    else:
                #        logging.debug('actually not solved!')
                #        reward -= 2
                #    self.mc.sendCommand("quit")
                #    self.agent.push_final(reward)
                #    break
                #elif solved:
                #    logging.debug('solved but not signaling about that')
                #    reward -= 2
                #    self.mc.sendCommand("quit")
                #    self.agent.push_final(reward)
                #    break
            logging.debug('reward %f', reward)
            new_actions = self.agent(data, reward=reward, epsilon=eps)
            data['action'] = self.agent.prev_action
            eps = max(eps * eps_decay, eps_end)
            logging.debug('epsilon %f', eps)
            self.state_queue.append(data)
            if t == max_t and reward <= 0:
                reward -= 1
                logging.debug("too long")
                stop_motion(mc)
                self.agent.push_final(reward)
                self.mc.sendCommand("quit")
                self.learn(self.agent, self.optimizer)
                break
            self.act(new_actions)
            time.sleep(0.4)
            stop_motion(mc)
            time.sleep(0.1)

            total_reward += reward
        # in termial state reward is not added due loop breaking
        total_reward += reward
        logging.info("Final reward: %f" % reward)

        mean_loss = numpy.mean([self.learn(self.agent, self.optimizer) for _ in range(3)])
        logging.info('loss %f', mean_loss)
        return total_reward, t, solved

    def is_solved(self, target):
        solved = False
        reward = -0.5
        if target is not None:
            if target[0] == 'log':
                reward = 45
                solved = True
            elif target[0] == 'leaves':
                aPos = self.mc.getAgentPos()
                coords = aPos[0], aPos[2]
                coords1 = target[1], target[-1]

                dist = numpy.linalg.norm(numpy.asarray(coords) - numpy.asarray(coords1), 2)
                logging.debug('dist %f', dist)
                if dist < 19:
                    solved = False
                    reward = 1
                else:
                    solved = True
                    reward = 40
        return solved, reward

    def act(self, actions):
        mc = self.mc
        for act in actions:
            logging.debug('action %s', act)
            if act == 'jump_forward':
                mc.sendCommand('move 0.4')
                mc.sendCommand('jump 1')
            else:
                mc.sendCommand(str(act))

    @classmethod
    def init_mission(cls, i, mc):
        miss = mb.MissionXML()
        video_producer = mb.VideoProducer(width=320 * 4, height=240 * 4, want_depth=cls.want_depth)

        obs = mb.Observations()
        agent_handlers = mb.AgentHandlers(observations=obs,
            all_str=mission_ending, video_producer=video_producer)

        center_x, center_y = -69.25, -23.14
        step = 30
        start_x = center_x + random.choice(numpy.arange(-step, step))
        start_y = center_y + random.choice(numpy.arange(-step, step))

        logging.info('starting at ({0}, {1})'.format(start_x, start_y))
        miss = mb.MissionXML(agentSections=[mb.AgentSection(name='Cristina',
                 agenthandlers=agent_handlers,
                                          #    depth
                 agentstart=mb.AgentStart([start_x, 30.0, start_y, 1]))])

        miss.setWorld(mb.flatworld("3;7,25*1,3*3,2;1;stronghold,biome_1,village,decoration,dungeon,lake,mineshaft,lava_lake",
            seed='43',
            forceReset="false"))
        miss.serverSection.initial_conditions.allowedmobs = "Pig Sheep Cow Chicken Ozelot Rabbit Villager"
        miss.serverSection.initial_conditions.time_pass = 'false'
        miss.serverSection.initial_conditions.time_start = "1000"


        if mc is None:
            mc = MCConnector(miss)
        else:
            mc.setMissionXML(miss)
        return mc

