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
from network import QVisualNetwork


class QVisualNetworkV2(QVisualNetwork):
    def __init__(self, n_prev_images, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_prev_images = n_prev_images
        num = kwargs.get('num', 128)
        # fully connected
        self.q_value = nn.Sequential(
            nn.Linear(num + (28 * 8 * 8 + 28 * 4 * 4 + 28) * 3, num),
            self.activation,
            nn.Linear(num, num),
            self.activation,
            nn.Linear(num, num),
            self.activation,
            nn.Linear(num, self.n_actions))


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
        angle_emb = common.angle_embed(state_data[:, :3])
        dist_emb = common.dist_embed(state_data[:, 3:6])
        state_data = torch.cat([angle_emb, dist_emb, state_data[:, 6:]], dim=1)
        state_emb = self.pos_emb(state_data)
        visual_pos_emb = torch.cat([visual_data.view(B, -1), state_emb], dim=1)
        result = self.q_value(visual_pos_emb)
        if torch.isnan(result).any().item():
            import pdb;pdb.set_trace()

        return result



class DeadException(RuntimeError):
    def __init__(self):
        super().__init__("it's dead")


# quit by reaching target or when zero health
mission_ending = """
<MissionQuitCommands quitDescription="give_up"/>
<RewardForMissionEnd>
  <Reward description="give_up" reward="243"/>
</RewardForMissionEnd>
"""


def visualize(yaw, dist):
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

    policy_net = QVisualNetworkV2(3, actionSet, 0, 41,  n_channels=3, activation=nn.LeakyReLU(), batchnorm=False, num=256)
    target_net = QVisualNetworkV2(3, actionSet, 0, 41,  n_channels=3, activation=nn.LeakyReLU(), batchnorm=False, num=256)
    batch_size = 18

    transformer = common.make_noisy_transformers()
    my_simple_agent = network.DQN(policy_net, target_net, 0.99, batch_size, 450, capacity=5000, transform=transformer)
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
        self.state_queue = deque(maxlen=2)
        if self.episode_stats:
            logging.info('average reward in episode stats {0}'.format(numpy.mean([v[0] for v in self.episode_stats.values()])))

    def _random_turn(self):
        turn = numpy.random.random() * random.choice([-1, 1])
        self.act(["turn {0}".format(turn)])
        self.act(["pitch {0}".format(0.1)])
        time.sleep(0.5)
        stop_motion(self.mc)

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

        # target
        pitch, yaw, dist = direction_to_target(mc, target_pos)
        # 'XPos', 'YPos', 'ZPos', 'Pitch', 'Yaw'
        # take pitch, yaw
        # take XPos, YPos, ZPos modulo 1
        self_pitch = normAngle(aPos[3] * math.pi/180.)
        self_yaw = normAngle(aPos[4] * math.pi/180.)
        xpos, ypos, zpos = aPos[0:3]
        logging.debug("%.2f %.2f %.2f ", xpos, ypos, zpos)
        # use relative height
        ypos = 30 - aPos[1]
        data = dict()


        img = img_data.reshape((240, 320, 3 + self.want_depth)).transpose(2, 0, 1) / 255.
        img = torch.as_tensor(img).float()
        data['image'] = img

        actions = []
        imgs = [torch.as_tensor(img)]
        yaws = [torch.as_tensor(yaw)]
        dists = [torch.as_tensor(dist)]
        heights = [torch.as_tensor(ypos)]
        # first prev, then prev_prev etc..
        for item in reversed(self.state_queue):
            actions.append(item['action'])
            imgs.append(item['image'])
            yaws.append(item['yaw'])
            dists.append(item['dist'])
            heights.append(item['ypos'])
        while len(imgs) < 3:
            imgs.append(img.to(img))
            actions.append(torch.as_tensor(-1).to(img))
            yaws.append(torch.as_tensor(yaw).to(img))
            dists.append(torch.as_tensor(dist).to(img))
            heights.append(torch.as_tensor(ypos).to(img))
        state = torch.as_tensor(yaws + dists + heights + actions)
        data.update(dict(state=state,
                         images=torch.stack(imgs),
                         image=img,
                         dist=torch.as_tensor(dist),
                         yaw=torch.as_tensor(yaw),
                         ypos=torch.as_tensor(ypos)
                         ))

        logging.debug('current_dist %i', dist)

        # depth
        coords1 = self.collect_visible(data, aPos[:3])
        if self.write_visualization:
            visualize(yaw, dist)
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
            return start_x, start_y
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
            return XPos, YPos

    @property
    def end(self):
        return self.target_x, self.target_y

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
        start = self._start()
        end = self.end
        max_t = 3000
        self._random_turn()

        logging.info('current target (%i, %i)', self.target_x, self.target_y)

        mc = self.mc
        logging.debug('memory: %i', self.agent.memory.position)
        self.agent.train()

        logging.info('max dist %i', max_t)
        eps_start = self.eps
        eps_end = 0.05
        eps_decay = 0.997

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
            except DeadException:
                stop_motion(mc)
                self.agent.push_final(-100)
                reward = -100
                logging.debug("failed at step %i", t)
                self.learn(self.agent, self.optimizer)
                break
            if self.state_queue:
                life = mc.getLife()
                logging.debug('current life %f', life)
                if life == 0:
                    reward = -100
                    stop_motion(mc)
                    if t > 2:
                        self.agent.push_final(reward)
                    self.learn(self.agent, self.optimizer)
                    break
                logging.debug('distance %f', data['dist'])
                prev_target_dist = self.state_queue[-1]['dist']
                dist_diff = (prev_target_dist - data['dist'])
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
                if data['dist'] < 0.58:
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
            logging.debug("current reward %f", reward)
            new_actions = self.agent(data, reward=reward, epsilon=eps)
            eps = max(eps * eps_decay, eps_end)
            logging.debug('epsilon %f', eps)
            data['action'] = self.agent.prev_action
            if 'visible' in data:
                data.pop('visible')
            self.state_queue.append(data)
            self.act(new_actions)
            time.sleep(0.4)
            stop_motion(mc)
            time.sleep(0.1)
            if t == max_t or total_reward < -200:
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

