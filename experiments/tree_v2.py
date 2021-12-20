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

import tagilmo.utils.mission_builder as mb
from tagilmo.utils.malmo_wrapper import MalmoConnector
import common
from common import stop_motion, grid_to_vec_walking, direction_to_target
from tagilmo.utils.mathutils import normAngle


mission_ending = """
<MissionQuitCommands quitDescription="give_up"/>
<RewardForMissionEnd>
  <Reward description="give_up" reward="243"/>
</RewardForMissionEnd>
"""


class DeadException(RuntimeError):
    def __init__(self):
        super().__init__("it's dead")


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

    policy_net = network.QVisualNetwork(actionSet, 2, 20, n_channels=3, activation=nn.ReLU(), batchnorm=True)
    target_net = network.QVisualNetwork(actionSet, 2, 20, n_channels=3, activation=nn.ReLU(), batchnorm=True)
    batch_size = 20
    my_simple_agent = network.DQN(policy_net, target_net, 0.9, batch_size, 450, capacity=2000)
    location = 'cuda' if torch.cuda.is_available() else 'cpu'
    if os.path.exists(path):
        logging.info('loading model from %s', path)
        data = torch.load(path, map_location=location)
        my_simple_agent.load_state_dict(data, strict=False)

    return my_simple_agent.to(location)


class Trainer(common.Trainer):
    want_depth = False

    def __init__(self, agent, mc, optimizer, eps, train=True):
        super().__init__(train)
        self.agent = agent
        self.mc = mc
        self.optimizer = optimizer
        self.eps = eps
        self.agent.to(self.device)

    def is_tree_visible(self):
        logging.debug(self.mc.getLineOfSight('type'))
        if self.mc.getLineOfSight('type') in ['log', 'leaves']:
            return [self.mc.getLineOfSight('type'),
                    self.mc.getLineOfSight('x'),
                    self.mc.getLineOfSight('y'),
                    self.mc.getLineOfSight('z')]
        return None

    def collect_state(self):
        while True:
            self.mc.observeProc()
            data = self.mc.getImage()
            aPos = self.mc.getAgentPos()
            if not any(x is None for x in (data, aPos)):
                self_pitch = normAngle(aPos[3]*math.pi/180.)
                self_yaw = normAngle(aPos[4]*math.pi/180.)

                data = data.reshape((240, 320, 3 + self.want_depth)).transpose(2, 0, 1) / 255.
                pitch_yaw = torch.as_tensor([self_pitch, self_yaw])
                return dict(image=torch.as_tensor(data).float(), position=pitch_yaw)
            else:
                time.sleep(0.05)

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

        max_t = 50
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
        target_vec = torch.zeros(20)
        current_target = random.choice([0, 1])
        possible_targets = 'log', 'water', 'sand', 'grass', 'leaves', 'water'
        current_target = random.choice(possible_targets)

        logging.info('current target %s', current_target)
        target_vec[common.visible_block_num[current_target]] = 1

        mean_loss = numpy.mean([self.learn(self.agent, self.optimizer) for _ in range(5)])
        logging.info('loss %f', mean_loss)
        while True:
            t += 1
            reward = 0
            try:
                data = self.collect_state()
                data['state'] = target_vec
                new_pos = data['position']
                target = self.is_tree_visible()
            except DeadException:
                stop_motion(mc)
                # should not die is this mission
                # so don't add this event to the replay buffer
                reward = 0
                logging.warning("died at step %i", t)
                break
            if prev_pos is None:
                prev_pos = new_pos
            else:
                # use only dist change for now
                life = mc.getLife()
                logging.debug('current life %f', life)
                if life == 0:
                    # should not die is this mission
                    continue
                reward += (life - prev_life) * 2
                prev_life = life
                if target is not None:
                    if common.visible_block_num[target[0]] == common.visible_block_num[current_target]:
                        reward = 100
                        self.agent.push_final(reward)
                        logging.debug('current target %s', target_str)
                        logging.debug('solved in %i steps', t)
                        mc.sendCommand("quit")
                        solved = True
                        break
                if reward == 0:
                    reward -= 1
            data['prev_pos'] = prev_pos
            logging.debug("current reward %f", reward)
            new_actions = self.agent(data, reward=reward, epsilon=eps)
            eps = max(eps * eps_decay, eps_end)
            logging.debug('epsilon %f', eps)
            self.act(new_actions)
            time.sleep(0.4)
            stop_motion(mc)
            time.sleep(0.1)
            prev_pos = new_pos
            if t == max_t:
                logging.debug("too long")
                stop_motion(mc)
                reward = -10
                self.agent.push_final(-10)
                self.mc.sendCommand("quit")
                self.learn(self.agent, self.optimizer)
                break
            total_reward += reward
        # in termial state reward is not added due loop breaking
        total_reward += reward
        logging.info("Final reward: %d" % reward)

        return total_reward, t, solved

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
        video_producer = mb.VideoProducer(width=320, height=240, want_depth=cls.want_depth)

        obs = mb.Observations()
        agent_handlers = mb.AgentHandlers(observations=obs,
            all_str=mission_ending, video_producer=video_producer)
        # a tree is at -18, 15
        # stay between tree and pond
        center_x = -23.5
        center_y = 12.5
        start_x = center_x + random.choice(numpy.arange(-2, 2))
        start_y = center_y + random.choice(numpy.arange(-5, 5))

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

