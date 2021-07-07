"""
cliff-walking environment and agent
"""
import logging
import random
import time
import torch
from torch import nn
import os
import network
import numpy
import math
from collections import deque, defaultdict

import tagilmo.utils.mission_builder as mb
from tagilmo.utils.malmo_wrapper import MalmoConnector
import common
from common import stop_motion, grid_to_vec_walking, \
    direction_to_target, normAngle


class DeadException(RuntimeError):
    def __init__(self):
        super().__init__("it's dead")


spiral = """
   <DrawLine x1="3"  y1="45" z1="1"  x2="8" y2="45" z2="2" type="sandstone" />
   <DrawLine x1="8"  y1="45" z1="2"  x2="10" y2="45" z2="4" type="sandstone" />         <!-- floor of the arena -->
   <DrawLine x1="10"  y1="45" z1="4"  x2="14" y2="45" z2="7" type="sandstone" />
   <DrawLine x1="14"  y1="45" z1="7"  x2="14" y2="45" z2="9" type="sandstone" />
   <DrawLine x1="14"  y1="45" z1="9"  x2="10" y2="45" z2="11" type="sandstone" />
   <DrawLine x1="10"  y1="45" z1="11"  x2="8" y2="45" z2="12" type="sandstone" />
   <DrawLine x1="8"  y1="45" z1="12"  x2="6" y2="45" z2="14" type="sandstone" />
   <DrawLine x1="6"  y1="45" z1="14"  x2="5" y2="45" z2="15" type="sandstone" />
   <DrawLine x1="5"  y1="45" z1="15"  x2="3" y2="45" z2="13" type="sandstone" />
"""

dec_xml = """
<DrawingDecorator>
        <!-- coordinates for cuboid are inclusive -->
        <DrawCuboid x1="-2" y1="46" z1="-2" x2="17" y2="80" z2="18" type="air" />            <!-- limits of our arena -->
        <DrawCuboid x1="-2" y1="45" z1="-2" x2="17" y2="45" z2="18" type="lava" />           <!-- lava floor -->
       {1}

       {0}
        <DrawBlock   x="4"   y="45"  z="1"  type="cobblestone" />                           <!-- the starting marker -->
        <DrawBlock   x="4"   y="45"  z="12" type="lapis_block" />                           <!-- the destination marker -->
        <DrawBlock   x="0"   y="45"  z="12" type="lapis_block" />
        <DrawItem    x="4"   y="46"  z="12" type="diamond" />
        <DrawItem    x="0"   y="46"  z="12" type="wooden_sword" />
</DrawingDecorator>
"""

modify_blocks = """
        <DrawLine x1="{0}"  y1="45" z1="5"  x2="4" y2="45" z2="0" type="sandstone"/>
        <DrawLine x1="15"  y1="46" z1="{2}"  x2="{1}" y2="46" z2="{2}" type="sandstone"/>
        <DrawLine x1="{0}"  y1="45" z1="5"  x2="4" y2="45" z2="13" type="sandstone"/>
"""

# quit by reaching target or when zero health
mission_ending = """
<MissionQuitCommands quitDescription="give_up"/>
<RewardForMissionEnd>
  <Reward description="give_up" reward="243"/>
</RewardForMissionEnd>
"""


def load_agent(path):
    # possible actions are
    # move[-1, 1],
    # strafe[-1, 1]
    # pitch[-1, 1]
    # turn[-1, 1]
    # jump 0/1

    # for example:
    # actionSet = [network.ContiniousAction('move', -1, 1),
    #              network.ContiniousAction('strafe', -1, 1),
    #              network.ContiniousAction('pitch', -1, 1),
    #              network.ContiniousAction('turn', -1, 1),
    #              network.BinaryAction('jump')]

    # discreet actions
    action_names = ["turn 0.1", "turn -0.1", "move 0.9", "jump_forward" ]
    actionSet = [network.CategoricalAction(action_names)]

    policy_net = network.QVisualNetwork(actionSet, 5, 3, n_channels=3, activation=nn.LeakyReLU(), batchnorm=True)
    target_net = network.QVisualNetwork(actionSet, 5, 3, n_channels=3, activation=nn.LeakyReLU(), batchnorm=True)
    batch_size = 28
    my_simple_agent = network.DQN(policy_net, target_net, 0.99, batch_size, 450, capacity=7000)
    location = 'cuda' if torch.cuda.is_available() else 'cpu'
    if os.path.exists(path):
        logging.info('loading model from %s', path)
        data = torch.load(path, map_location=location)
        my_simple_agent.load_state_dict(data, strict=False)

    return my_simple_agent.to(location)


def iterative_avg(current, new):
    return current + 0.01 * (new - current)


def inverse_priority_sample(weights: numpy.array):
    weights = weights - weights.min() + 0.0001
    r = numpy.random.random(len(weights))
    w_new = r ** (1 / weights)
    idx = numpy.argmin(w_new)
    return idx


class Trainer(common.Trainer):
    want_depth = False

    def __init__(self, agent, mc, optimizer, eps, train=True):
        super().__init__(train)
        self.from_queue = False
        self.write_visualization = False
        self.agent = agent
        self.mc = mc
        self.optimizer = optimizer
        if random.random() < 0.2:
            eps = 0.05
        logging.info('start eps %f', eps)
        self.eps = eps
        self.target_x = 4.5
        self.target_y = 12.5
        self.dist = max(5, round(abs(numpy.random.normal()) * 110))
        self.episode_stats = agent.memory.episode_stats
        self.failed_queue = agent.memory.failed_queue
        self.img_num = 0
        if self.episode_stats:
            logging.info('average reward in episode stats {0}'.format(numpy.mean([v[0] for v in self.episode_stats.values()])))

    def _random_turn(self):
        turn = numpy.random.random() * random.choice([-1, 1])
        self.act(["turn {0}".format(turn)])
        time.sleep(0.5)
        stop_motion(self.mc)

    def collect_state(self):
        mc = self.mc
        target =  ['lapis_block', self.target_x, 30, self.target_y]
        target_pos = target[1:4]
        mc.observeProc()
        aPos = mc.getAgentPos()
        img_data = self.mc.getImage()
        logging.debug(aPos)
        while aPos is None or (img_data is None):
            time.sleep(0.05)
            mc.observeProc()
            aPos = mc.getAgentPos()
            img_data = self.mc.getImage()
            if not all(mc.isAlive):
                raise DeadException()
        # grid
        grid = mc.getNearGrid()

        grid_vec = grid_to_vec_walking(grid[:36])
        # position encoding
        grid_enc = torch.as_tensor(grid_vec)
        # target
        pitch, yaw, dist = direction_to_target(mc, target_pos)
        target_enc = torch.as_tensor([pitch, yaw, dist])
        # 'XPos', 'YPos', 'ZPos', 'Pitch', 'Yaw'
        # take pitch, yaw
        # take XPos, YPos, ZPos modulo 1
        self_pitch = normAngle(aPos[3] * math.pi/180.)
        self_yaw = normAngle(aPos[4] * math.pi/180.)
        xpos, ypos, zpos = [_ % 1 for _ in aPos[0:3]]
        # use relative height
        ypos = 30 - aPos[1]
        logging.debug("%.2f %.2f %.2f ", xpos, ypos, zpos)
        self_pos_enc = torch.as_tensor([self_pitch, self_yaw, xpos, ypos, zpos])
        data = dict(grid_vec=grid_enc, target=target_enc, state=target_enc, pos=self_pos_enc)

        img = img_data.reshape((240, 320, 3 + self.want_depth)).transpose(2, 0, 1) / 255.
        data['image'] = torch.as_tensor(img).float()
        # depth
        visible = self.mc.getLineOfSight('type')
        if visible is not None:
            coords = [self.mc.getLineOfSight('x'),
                      self.mc.getLineOfSight('y'),
                      self.mc.getLineOfSight('z')]
            height = 1.6025
            coords1 = aPos[:3]
            coords1[1] += height
            dist = numpy.linalg.norm(numpy.asarray(coords) - numpy.asarray(coords1), 2)
            data['visible'] = [visible] + coords + [dist]
        if self.write_visualization:
            import cv2
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (40,40)
            fontScale              = 0.9
            fontColor              = (15, 15, 15)
            lineType               = 2
            img_draw = (img * 255).astype(numpy.uint8)
            img_draw = cv2.putText(img_draw.transpose(1,2,0), 'distance {0:.1f}'.format(dist),bottomLeftCornerOfText, font,fontScale,fontColor,lineType)
            #img_draw = cv2.putText(img_draw, 'yaw {0:.1f}'.format(yaw),
            #                       (10, 80),
            #                       font,fontScale,fontColor,lineType)
            c_x = 260
            c_y = 200
            r = 20
            img_draw = cv2.circle(img_draw,
                                  (c_x, c_y),
                                  r,
                                  (0,255,255), 2)
            cos_x = numpy.cos(yaw + numpy.pi / 2) * r
            sin_y = numpy.sin(yaw + numpy.pi / 2) * r
            img_draw = cv2.line(img_draw,
                               (c_x, c_y),
                               (round(c_x - cos_x),
                                round(c_y - sin_y)), (0, 255, 255), 2)
            cv2.imwrite('episodes/img{0}.png'.format(self.img_num), img_draw)
            self.img_num += 1
            #cv2.imshow('1', img_draw)
            #cv2.waitKey(100)

        #if self.want_depth:
        #    depth = data['image'][-1]
        #    h, w = [_ // 2 for _ in depth.shape]
        #    img_depth = img[-1][h, w]
        #    norm_depth = (depth * (dist / img_depth))
        #    data['image'][-1] = norm_depth
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                assert not torch.isnan(value).any()
        return data

    def _start(self):
        if random.random() < 0.6 and (self.failed_queue or self.episode_stats):
            self.mc.sendCommand("quit")
            time.sleep(1)
            if self.failed_queue:
                self.from_queue = True
                start, end = self.failed_queue.pop()
                x, y = end
                start_x = x + random.choice((-15, 15))
                start_y = y + random.choice((-15, 15))
                logging.info('evaluating from queue {0}, {1}'.format(start, end))
            else:
                pairs = list(self.episode_stats.items())
                r = numpy.asarray([p[1][0] for p in pairs])
                idx = inverse_priority_sample(r)
                logging.debug('prority sample idx=%i', idx)
                start, end = pairs[idx][0]
                logging.info('evaluating from stats {0}, {1}'.format(start, end))
                start_x, start_y = start
            # start somewhere near end
            self.mc = self.init_mission(0, self.mc, start_x=start_x, start_y=start_y)
            self.mc.safeStart()
            self.target_x, self.target_y = end
        else:
            self.mc.observeProc()
            aPos = self.mc.getAgentPos()
            while aPos is None:
                time.sleep(0.05)
                self.mc.observeProc()
                aPos = self.mc.getAgentPos()

            XPos, _, YPos = aPos[:3]
            self.target_x = XPos + random.choice(numpy.arange(-self.dist, self.dist))
            self.target_y = YPos + random.choice(numpy.arange(-self.dist, self.dist))

            start = (XPos, YPos)
            end = self.target_x, self.target_y
        return start, end

    def _end(self, start, end, solved, t, total_reward):
        if self.from_queue:

            """
            If episode failed check length, if 15 < t
                start from a different point near the target
                if success add to episode_stats
            """
            # failed
            if not solved:
                pass
            else:
                logging.info('from queue run succeeded')
                self.episode_stats[(start, end)] = [0, 0]
        else:
            if (start, end) in self.episode_stats:
                r, l = self.episode_stats[(start, end)]
                logging.info('old stats ({0}, {1})'.format(r, l))
                r = iterative_avg(r, total_reward)
                l = iterative_avg(l, t)
                self.episode_stats[(start, end)] = (r, l)
                logging.info('new episode stats reward {0} length {1}'.format(r, l))
            elif 10 < t and not solved:
                self.failed_queue.append((start, end))
                logging.info('adding to failed queue {0}, {1}'.format(start, end))
        if random.random() < 0.2:
            mean_loss = numpy.mean([self.learn(self.agent, self.optimizer) for _ in range(20)])
            logging.info('loss %f', mean_loss)

    def run_episode(self):
        """ Deep Q-Learning episode
        """
        self.agent.clear_state()
        start, end = self._start()
        self._random_turn()
        logging.info('current target (%i, %i)', self.target_x, self.target_y)

        mc = self.mc
        logging.debug('memory: %i', self.agent.memory.position)
        self.agent.train()
        d = round(numpy.linalg.norm(numpy.array(end) - start, 2))
        logging.info('current_dist %i', d)
        max_t = d ** 2 + 50
        logging.info('max dist %i', max_t)
        eps_start = self.eps
        eps_end = 0.05
        eps_decay = 0.998

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
            # target = search4blocks(mc, ['lapis_block'], run=False)
            reward = 0
            try:
                data = self.collect_state()
                target_enc = data['target']
                new_pos = data['pos']
            except DeadException:
                stop_motion(mc)
                self.agent.push_final(-100)
                reward = -100
                logging.debug("failed at step %i", t)
                self.learn(self.agent, self.optimizer)
                break
            if prev_pos is None:
                prev_pos = new_pos
            else:
                life = mc.getLife()
                logging.debug('current life %f', life)
                if life == 0:
                    reward = -100
                    stop_motion(mc)
                    if t > 2:
                        self.agent.push_final(reward)
                    self.learn(self.agent, self.optimizer)
                    break
                logging.debug('distance %f', target_enc[2])
                dist_diff = (prev_target_dist - target_enc)[2]
                if dist_diff > 1:
                    reward += 1
                if dist_diff < 0:
                    reward -= 1
                reward += dist_diff + (life - prev_life) * 2
                prev_life = life
                grid = mc.getNearGrid()
                if not mc.is_mission_running():
                    logging.debug('failed in %i steps', t)
                    reward = -100
                if target_enc[2] < 0.58:
                    time.sleep(1)
                    mc.observeProc()
                    life = mc.getLife()
                    mc.sendCommand("quit")
                    if life == prev_life:
                        reward += 25
                        self.agent.push_final(reward)
                    logging.debug('solved in %i steps', t)
                    solved = True
                    break
            if reward == 0:
                reward -= 0.5
                if 'visible' in data:
                    d = data['visible'][-1]
                    if d < 1:
                        logging.debug('visible {0}'.format(d))
                        reward -= 3
                # median_depth = numpy.median(data['image'][-1])
                # if median_depth < 2:
                #    reward -= 1 / median_depth
            if 'visible' in data:
                data.pop('visible')
            logging.debug("current reward %f", reward)
            data['prev_pos'] = prev_pos
            data['position'] = data['pos']
            new_actions = self.agent(data, reward=reward, epsilon=eps)
            eps = max(eps * eps_decay, eps_end)
            logging.debug('epsilon %f', eps)
            self.act(new_actions)
            time.sleep(0.4)
            stop_motion(mc)
            time.sleep(0.1)
            prev_pos = new_pos
            prev_target_dist = target_enc
            if t == max_t or total_reward < -120:
                reward -= 1
                logging.debug("too long")
                stop_motion(mc)
                self.agent.push_final(reward)
                self.mc.sendCommand("quit")
                self.learn(self.agent, self.optimizer)
                break
            total_reward += reward
        # in termial state reward is not added due loop breaking
        total_reward += reward
        logging.debug("Final reward: %f", reward)
        self._end(start, end, solved, t, total_reward)
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
    def init_mission(cls, i, mc, start_x=None, start_y=None):
        miss = mb.MissionXML()
        video_producer = mb.VideoProducer(width=320, height=240, want_depth=cls.want_depth)

        obs = mb.Observations()

        obs = mb.Observations()
        obs.gridNear = [[-1, 1], [-2, 1], [-1, 1]]


        agent_handlers = mb.AgentHandlers(observations=obs,
            all_str=mission_ending)


        agent_handlers = mb.AgentHandlers(observations=obs,
            all_str=mission_ending, video_producer=video_producer)
        # a tree is at -18, 15
        if start_x is None:
            center_x = -18
            center_y = 15

            start_x = center_x + random.choice(numpy.arange(-329, 329))
            start_y = center_y + random.choice(numpy.arange(-329, 329))

        logging.info('starting at ({0}, {1})'.format(start_x, start_y))

        miss = mb.MissionXML(agentSections=[mb.AgentSection(name='Cristina',
                 agenthandlers=agent_handlers,
                                          #    depth
                 agentstart=mb.AgentStart([start_x, 30.0, start_y, 1]))])

        miss.setWorld(mb.flatworld("3;7,25*1,3*3,2;1;stronghold,biome_1,village,decoration,dungeon,lake,mineshaft,lava_lake",
            seed='43',
            forceReset="false"))
        miss.serverSection.initial_conditions.allowedmobs = "Pig Sheep Cow Chicken Ozelot Rabbit Villager"
        # uncomment to disable passage of time:
        miss.serverSection.initial_conditions.time_pass = 'false'
        miss.serverSection.initial_conditions.time_start = "1000"

        if mc is None:
            mc = MalmoConnector(miss)
        else:
            mc.setMissionXML(miss)
        return mc

