from operator import itemgetter
from log import setup_logger
import logging
import numpy
import network
import torch
import MalmoPython
import json
import logging
import math
import os
import random
import sys
import time
import numpy as np
from tagilmo.utils.malmo_wrapper import MalmoConnector
import tagilmo.utils.mission_builder as mb
from common import *

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

obs = mb.Observations()
obs.gridNear = [[-1, 1], [-2, 2], [-1, 1]]

current_xml = dec_xml.format(modify_blocks.format(4, -3, 8), '')
handlers = mb.ServerHandlers(mb.flatworld("3;7,220*1,5*3,2;3;,biome_1"), alldecorators_xml=current_xml, bQuitAnyAgent=True)
agent_handlers = mb.AgentHandlers(observations=obs, all_str=mission_ending)

miss = mb.MissionXML(serverSection=mb.ServerSection(handlers), agentSections=[mb.AgentSection(name='Cristina',
                     agenthandlers=agent_handlers,
                     agentstart=mb.AgentStart([4.5, 46.0, 1.5, 30]))])


mc = MalmoConnector(miss)
my_mission = mc.mission
my_mission.allowAllDiscreteMovementCommands()
#my_mission.requestVideo(320, 240)
my_mission.setViewpoint(1)


max_retries = 3
my_mission_record = mc.mission_record
#mc.safeStart()

agent_host = mc.agent_hosts[0]


class DeadException(RuntimeError):
    def __init__(self):
        super().__init__("it's dead")


def collect_state(mc, target_pos):
    mc.observeProc()
    aPos = mc.getAgentPos()
    logging.debug(aPos)
    while aPos is None:
        sleep(0.1)
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
    return grid_enc, target_enc, self_pos_enc


def action_state_to_vec(action_state):
    return torch.as_tensor([x.value for x in action_state])


def act(actions, mc):
    for act in actions:
        if act == 'jump_forward':
            mc.sendCommand('move 0.4')
            mc.sendCommand('jump 1')
        else:
            mc.sendCommand(str(act))


def stop_motion(mc):
    mc.sendCommand('move 0')
    mc.sendCommand('strafe 0')
    mc.sendCommand('pitch 0')
    mc.sendCommand('turn 0')
    mc.sendCommand('jump 0')


def learn(agent, optimizer):
    losses = []
    for i in range(40):
        optimizer.zero_grad()
        loss = agent.compute_loss()
        if loss is not None:
            # Optimize the model
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 2)
            optimizer.step()
            losses.append(loss.cpu().detach())
    if losses:
        logging.debug('optimizing')
        logging.debug('loss %f', numpy.mean(losses))

    return numpy.mean(losses)


def run_episode(agent, agent_host, eps, mc, optimizer):
    logging.debug('memory: %i', agent.memory.position)
    agent.train()

    time.sleep(0.05)
    max_t = 80
    eps_start = eps
    eps_end = 0.05
    eps_decay = 0.99

    # Deep Q-Learning
    #
    # Params
    # ======
    #     n_episodes (int): maximum number of training epsiodes
    #     max_t (int): maximum number of timesteps per episode
    #     eps_start (float): starting value of epsilon, for epsilon-greedy action selection
    #     eps_end (float): minimum value of epsilon
    #     eps_decay (float): mutiplicative factor (per episode) for decreasing epsilon

    scores = []  # list containing score from each episode
    # scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start

    is_first_action = True
    total_reward = 0

    # obs = json.loads(world_state.observations[-1].text)
    # if 'XPos' not in obs:
    #     logging.debug("skipping")
    #     return 0, 0
    # curr_x = obs[u'XPos']
    # curr_z = obs[u'ZPos']
    states = {}
    score = 0
    t = 0

    world_state = agent_host.getWorldState()
    # pitch, yaw, xpos, ypos, zpos
    prev_pos = None
    prev_target_dist = None
    prev_life = 20
    solved = False

    target =  ['lapis_block', 4.5, 46, 12.5]
    running = True
    mean_loss = numpy.mean([learn(agent, optimizer) for _ in range(5)])
    logging.info('loss %f', mean_loss)
    while running:
        t += 1
        # target = search4blocks(mc, ['lapis_block'], run=False)
        reward = 0
        logging.debug('\n\n')
        pos = target[1:4]
        try:
            grid_enc, target_enc, new_pos = collect_state(mc, pos)
        except DeadException:
            stop_motion(mc)
            agent.push_final(-100)
            reward = -100
            learn(agent, optimizer)
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
                agent.push_final(reward)
                running = False
                learn(agent, optimizer)
                break
            reward += (prev_target_dist - target_enc)[2] + (life - prev_life)
            if (prev_target_dist - target_enc)[2] <= -0.9:
                reward -= 10
            prev_life = life
            grid = mc.getNearGrid()
            if target_enc[2] < 0.53:
                reward = 100
                agent.push_final(reward)
                running = False
                mc.sendCommand("quit")
                solved = True
                break
            world_state = agent_host.getWorldState()
            if not world_state.is_mission_running:
                running = False
                reward = -100
                agent.push_final(reward)
                break
            if reward == 0:
                reward -= 2.5
            logging.debug("current reward %f", reward)
            if not world_state.is_mission_running:
                break
        data = dict(grid_vec=grid_enc, target=target_enc,
                pos=new_pos)
        new_actions = agent(data, reward=reward, epsilon=eps)
        eps = max(eps * eps_decay, eps_end)
        logging.debug('epsilon %f', eps)
        act(new_actions, mc)
        time.sleep(0.4)
        stop_motion(mc)
        time.sleep(0.1)
        prev_pos = new_pos
        prev_target_dist = target_enc
        world_state = agent_host.getWorldState()
        if t == max_t:
            logging.debug("too long")
            stop_motion(mc)
            reward = -10
            agent.push_final(-10)
            mc.sendCommand("quit")
            learn(agent, optimizer)
            break
        total_reward += reward
    logging.debug("Final reward: %d" % reward)

    return total_reward, t, solved


def simple_trainable_agent_test_remastered():
    # possible actions are
    # move[-1, 1],
    # strafe[-1, 1]
    # pitch[-1, 1]
    # turn[-1, 1]
    # jump 0/1
    actionSet = [network.ContiniousAction('move', -1, 1),
                 network.ContiniousAction('strafe', -1, 1),
                 network.ContiniousAction('pitch', -1, 1),
                 network.ContiniousAction('turn', -1, 1),
                 network.BinaryAction('jump')]

    # discreet actions
    # "move -0.5" "jump_forward",
    action_names = ["turn 0.15", "turn -0.15", "move 0.5", "jump_forward" ]
    actionSet = [network.CategoricalAction(action_names)]

    my_simple_agent = network.DQN(0.9, 70, 450, actionSet, capacity=2000,
                                                    grid_len=27, grid_w=5,
                                                    target_enc_len=3,
                                                    pos_enc_len=5)
    if os.path.exists('agent.pth'):
        data = torch.load('agent.pth')
        #name = 'action_output.6.weight'
        #data.pop(name)
        #data.pop('action_output.6.bias')
        my_simple_agent.load_state_dict(data, strict=False)
    num_repeats = 400
    comulative_rewards = []
    eps = 0.26
    eps_start = eps
    eps_end = 0.09
    eps_decay = 0.99
    optimizer = torch.optim.AdamW(my_simple_agent.parameters(), lr=0.0005,
                                    weight_decay=0.01)
    p = 1
    for i in range(0, num_repeats):
        sp = ''
        # train on simple environment first
        if i < 200:
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
        agent_handlers = mb.AgentHandlers(observations=obs, all_str=mission_ending)

        miss = mb.MissionXML(serverSection=mb.ServerSection(handlers), agentSections=[mb.AgentSection(name='Cristina',
                 agenthandlers=agent_handlers,
                 agentstart=mb.AgentStart([4.5, 46.0, 1.5, 30]))])
        mc.setMissionXML(miss)
        my_simple_agent.clear_state()


        logging.debug("\nMission %d of %d:" % (i + 1, num_repeats))
        mc.safeStart()

        # -- run the agent in the world -- #
        comulative_reward, steps, solved = run_episode(my_simple_agent, agent_host, eps, mc, optimizer)
        logging.info('episode %i: solved %i: comulative reward: %f', i, solved, comulative_reward)
        logging.debug("eps: %f", eps)
        comulative_rewards += [comulative_reward]
        eps = max(eps * eps_decay, eps_end)

        # -- clean up -- #
        time.sleep(0.5)  # (let the Mod reset)

        if i % 14 == 0:
            torch.save(my_simple_agent.state_dict(), 'agent.pth')

setup_logger('train.log')
simple_trainable_agent_test_remastered()
