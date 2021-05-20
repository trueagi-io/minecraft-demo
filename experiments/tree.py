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
from common import learn, stop_motion, grid_to_vec_walking, \
    direction_to_target, normAngle


mission_ending = """
<MissionQuitCommands quitDescription="give_up"/>
<RewardForMissionEnd>
  <Reward description="give_up" reward="243"/>
</RewardForMissionEnd>
"""


class DeadException(RuntimeError):
    def __init__(self):
        super().__init__("it's dead")


def load_agent_tree(path):
    # possible actions are
    # move[-1, 1],
    # strafe[-1, 1]
    # pitch[-1, 1]
    # turn[-1, 1]
    # jump 0/1

    # discreet actions
    # "move -0.5" "jump_forward",
    action_names = ["turn 0.15", "turn -0.15", "turn 0.05",
                   "turn 0.05", 'pitch 0.1', 'pitch -0.1']
    actionSet = [network.CategoricalAction(action_names)]

    policy_net = network.QVisualNetwork(actionSet, 2, n_channels=3, activation=nn.ReLU(), batchnorm=True)
    target_net = network.QVisualNetwork(actionSet, 2, n_channels=3, activation=nn.ReLU(), batchnorm=True)
    batch_size = 20
    my_simple_agent = network.DQN(policy_net, target_net, 0.9, batch_size, 450, capacity=2000)

    if os.path.exists('agent_tree.pth'):
        data = torch.load('agent_tree.pth')
        #name = 'action_output.6.weight'
        #data.pop(name)
        #data.pop('action_output.6.bias')
        my_simple_agent.load_state_dict(data, strict=False)

    return my_simple_agent



class Trainer:
    def __init__(self, agent, mc, optimizer, eps):
        self.agent = agent
        self.mc = mc
        self.optimizer = optimizer
        self.eps = eps
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.agent.to(self.device)
        logging.info('using device {0}'.format(self.device))

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

                data = data.reshape((240, 320, 3)).transpose(2, 0, 1) / 255.
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

        mean_loss = numpy.mean([learn(self.agent, self.optimizer) for _ in range(5)])
        logging.info('loss %f', mean_loss)
        while True:
            t += 1
            reward = 0
            try:
                data = self.collect_state()
                new_pos = data['position']
                target = self.is_tree_visible()
            except DeadException:
                stop_motion(mc)
                self.agent.push_final(-100)
                reward = -100
                logging.debug("failed at step %i", t)
                learn(self.agent, self.optimizer)
                break
            if prev_pos is None:
                prev_pos = new_pos
            else:
                # use only dist change for now
                life = mc.getLife()
                logging.debug('current life %f', life)
                if life == 0:
                    reward = -100
                    stop_motion(mc)
                    self.agent.push_final(reward)
                    learn(self.agent, self.optimizer)
                    break
                reward += (life - prev_life) * 2
                prev_life = life
                if target is not None:
                    if target[0] == 'log':
                        reward = 100
                        self.agent.push_final(reward)
                        logging.debug('solved in %i steps', t)
                        mc.sendCommand("quit")
                        solved = True
                        break
                    elif target[0] == 'leaves':
                        reward = 1
                if not mc.is_mission_running():
                    logging.info('failed in %i steps', t)
                    reward = -100
                    self.agent.push_final(reward)
                    break
                if reward == 0:
                    reward -= 2
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
                learn(self.agent, self.optimizer)
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

    @staticmethod
    def init_mission(i, mc):
        miss = mb.MissionXML()
        video_producer = mb.VideoProducer(width=320, height=240)

        obs = mb.Observations()
        agent_handlers = mb.AgentHandlers(observations=obs,
            all_str=mission_ending, video_producer=video_producer)
        # a tree is at -18, 15
        start_x = random.choice(numpy.arange(-22, -8))
        start_y = random.choice(numpy.arange(7, 21))
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

