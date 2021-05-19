import logging
import random
import time
import torch
import os
import network
import numpy
import math

import tagilmo.utils.mission_builder as mb
from tagilmo.utils.malmo_wrapper import MalmoConnector
from common import learn, stop_motion, grid_to_vec_walking, \
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


def load_agent_cliff(path):
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
    action_names = ["turn 0.15", "turn -0.15", "move 0.5", "jump_forward" ]
    actionSet = [network.CategoricalAction(action_names)]

    policy_net = network.QNetwork(actionSet, grid_len=27, grid_w=5, target_enc_len=3, pos_enc_len=5)
    target_net = network.QNetwork(actionSet, grid_len=27, grid_w=5, target_enc_len=3, pos_enc_len=5)
 
    my_simple_agent = network.DQN(policy_net, target_net, 0.9, 70, 450, capacity=2000)
    if os.path.exists(path):
        data = torch.load(path)
        my_simple_agent.load_state_dict(data, strict=False)

    return my_simple_agent



class Trainer:
    def __init__(self, agent, mc, optimizer, eps):
        self.agent = agent
        self.mc = mc
        self.optimizer = optimizer
        self.eps = eps

    def collect_state(self):
        mc = self.mc
        target =  ['lapis_block', 4.5, 46, 12.5]
        target_pos = target[1:4]
        mc.observeProc()
        aPos = mc.getAgentPos()
        logging.debug(aPos)
        while aPos is None:
            time.sleep(0.05)
            mc.observeProc()
            aPos = mc.getAgentPos()
            if not all(mc.isAlive):
                raise DeadException()
        # grid
        grid = mc.getNearGrid()
        grid_vec = grid_to_vec_walking(grid[:27])
        # position encoding
        grid_enc = torch.as_tensor(grid_vec)
        # target
        pitch, yaw, dist = direction_to_target(mc, target_pos)
        target_enc = torch.as_tensor([pitch, yaw, dist])
        # 'XPos', 'YPos', 'ZPos', 'Pitch', 'Yaw'
        # take pitch, yaw
        # take XPos, YPos, ZPos modulo 1
        self_pitch = normAngle(aPos[3]*math.pi/180.)
        self_yaw = normAngle(aPos[4]*math.pi/180.)
        xpos, ypos, zpos = [_ % 1 for _ in aPos[0:3]]
        logging.debug("%.2f %.2f %.2f ", xpos, ypos, zpos)
        self_pos_enc = torch.as_tensor([self_pitch, self_yaw, xpos, ypos, zpos])
        data = dict(grid_vec=grid_enc, target=target_enc, pos=self_pos_enc)
        return data

    def run_episode(self):
        """ Deep Q-Learning episode
        """
        self.agent.clear_state()
        mc = self.mc 
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
    
        mean_loss = numpy.mean([learn(self.agent, self.optimizer) for _ in range(5)])
        logging.info('loss %f', mean_loss)
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
                reward += (prev_target_dist - target_enc)[2] + (life - prev_life) * 2
                prev_life = life
                grid = mc.getNearGrid()
                if target_enc[2] < 0.53:
                    reward = 100
                    self.agent.push_final(reward)
                    logging.debug('solved in %i steps', t)
                    mc.sendCommand("quit")
                    solved = True
                    break
                if not mc.is_mission_running():
                    logging.debug('failed in %i steps', t)
                    reward = -100
                    self.agent.push_final(reward)
                    break
                if reward == 0:
                    reward -= 2
            logging.debug("current reward %f", reward)
            new_actions = self.agent(data, reward=reward, epsilon=eps)
            eps = max(eps * eps_decay, eps_end)
            logging.debug('epsilon %f', eps)
            self.act(new_actions)
            time.sleep(0.4)
            stop_motion(mc)
            time.sleep(0.1)
            prev_pos = new_pos
            prev_target_dist = target_enc
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
        logging.debug("Final reward: %d" % reward)
    
        return total_reward, t, solved


    def act(self, actions):
        mc = self.mc
        for act in actions:
            if act == 'jump_forward':
                mc.sendCommand('move 0.4')
                mc.sendCommand('jump 1')
            else:
                mc.sendCommand(str(act))
   
    @staticmethod
    def init_mission(i, mc):
        sp = ''
        # train on simple environment first
        if i < 10:
            p = random.choice([x for x in range(2, 8)])
        else:
            p = random.choice([-2, -1] + [x for x in range(0, 12)])
            if random.choice([True, False]):
                sp = spiral
        jump_block =  -3
        jump_block1 = random.choice(list(range(2, 11)))
        logging.debug('%i, %i', jump_block, jump_block1)
        current_xml = dec_xml.format(modify_blocks.format(p, jump_block, jump_block1), sp)
        handlers = mb.ServerHandlers(mb.flatworld("3;7,220*1,5*3,2;3;,biome_1"), alldecorators_xml=current_xml, bQuitAnyAgent=True)
        video_producer = mb.VideoProducer(width=320, height=240)
    
        obs = mb.Observations()
        obs.gridNear = [[-1, 1], [-2, 2], [-1, 1]]
    
    
        agent_handlers = mb.AgentHandlers(observations=obs,
            all_str=mission_ending)
    
        miss = mb.MissionXML(serverSection=mb.ServerSection(handlers), agentSections=[mb.AgentSection(name='Cristina',
                 agenthandlers=agent_handlers,
                 agentstart=mb.AgentStart([4.5, 46.0, 1.5, 30]))])
        if mc is None:
            mc = MalmoConnector(miss)
        else:
            mc.setMissionXML(miss)
        return mc 
