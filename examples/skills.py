import torch
import logging
from time import sleep, time
import math
from random import random

from tagilmo.utils.vereya_wrapper import MCConnector, RobustObserver
import tagilmo.utils.mission_builder as mb
from tagilmo.utils.mathutils import *

from examples.goal import *



class LookPitch(RobGoal):

    def __init__(self, rob, pitch, speed = 0.4):
        super().__init__(rob)
        self.dPitch = None
        self.update_target(pitch)
        self.speed = speed

    def update(self):
        aPos = self.rob.cached['getAgentPos'][0]
        if self.pitch is not None and aPos is not None:
            self.dPitch = normAngle(self.pitch - degree2rad(aPos[3]))

    def act(self):
        if self.dPitch is not None:
            return [["pitch", str(self.dPitch * self.speed)]]
        return []

    def finished(self):
        if self.dPitch is not None:
            return abs(self.dPitch) < 0.02
        else:
            return None

    def stop(self):
        return [["pitch", "0"]]

    def update_target(self, pitch):
        self.pitch = pitch


class LookYaw(RobGoal):

    def __init__(self, rob, yaw, speed=0.4):
        super().__init__(rob)
        self.dYaw = None
        self.update_target(yaw)
        self.speed = speed

    def update(self):
        aPos = self.rob.cached['getAgentPos'][0]
        if self.yaw is not None and aPos is not None:
            self.dYaw = normAngle(self.yaw - degree2rad(aPos[4]))

    def act(self):
        if self.dYaw is not None:
            return [["turn", str(self.dYaw * self.speed)]]
        return []

    def finished(self):
        if self.dYaw is not None:
            return abs(self.dYaw) < 0.02
        else:
            return None

    def stop(self):
        return [["turn", "0"]]

    def update_target(self, yaw):
        self.yaw = yaw


class LookDir(RobGoal):

    def __init__(self, rob, pitch, yaw):
        super().__init__(rob,
            CAnd([LookPitch(rob, pitch), LookYaw(rob, yaw)]))

    def update_target(self, pitch, yaw):
        # TODO: automate? generic update_target with one argument
        self.delegate.subgoals[0].update_target(pitch)
        self.delegate.subgoals[1].update_target(yaw)

    def set_speed(self, speed):
        for g in self.delegate.subgoals:
            g.speed = speed


class LookAt(RobGoal):

    def __init__(self, rob, pos):
        super().__init__(rob,
            LookDir(rob, 0, 0))
        self.update_target(pos)

    def update(self):
        pitch, yaw = self.rob.dirToAgentPos(self.target_pos, False)
        self.delegate.update_target(pitch, yaw)
        super().update()

    def update_target(self, pos):
        self.target_pos = pos

    def set_speed(self, speed):
        self.delegate.set_speed(speed)


class AttackInRange(RobGoal):

    def __init__(self, rob, hitType=['BLOCK']):
        super().__init__(rob)
        self.target = None
        self.hitType = hitType

    def update(self):
        los = self.rob.cached['getLineOfSights'][0]
        self.target = None if (los is None or los['hitType'] is None or \
            los['hitType'] not in self.hitType or not los['inRange']) else los

    def act(self):
        return [] if self.finished() else [['attack', '1']]

    def stop(self):
        return [['attack', '0']]

    def finished(self):
        return self.target is None


class AttackBlockSight(RobGoal):

    '''
    Attack one (hopefully) block at sight.
    No tool choice. Entity from mining is not guaranteed.
    The purpose is to destroy the block.
    '''

    def __init__(self, rob):
        super().__init__(rob,
            AttackInRange(rob))
        self.target = None

    def __get_target(self):
        t = self.delegate.target
        return None if t is None else int_coords([t['x'], t['y'], t['z']])

    def finished(self):
        #TODO ? it seems that sometimes it continues to attack one more block which spoils ladders
        if super().finished():
            return True
        cur_t = self.__get_target()
        if self.target is None:
            self.target = cur_t
            aPos = self.rob.cached['getAgentPos'][0]
            # proper distance to the block center
            self.max_dist = dist_vec([aPos[0], aPos[1]+1.66, aPos[2]],
                                     self.rob.blockCenterFromPos(cur_t))
            return False
        cur_dist = self.delegate.target['distance']
        return self.target != cur_t and \
            (cur_dist > self.max_dist or dist_vec(self.target, cur_t) > self.max_dist)
            


class AttackBlockTool(RobGoal):

    '''
    Try to select tool for mining if available before attacking,
    but doesn't try to make the tool required for mining
    '''
    def __init__(self, rob):
        super().__init__(rob,
            CAnd([SelectMineTool(rob), AttackBlockSight(rob)]))



class LookAndAttackBlock(RobGoal):

    def __init__(self, rob, pos):
        pos = rob.blockCenterFromPos(pos)
        look = LookAt(rob, pos)
        attack = AttackBlockTool(rob)
        super().__init__(rob,
            # SAnd([look, CAnd([look, attack])]))
            SAnd([look, attack]))


class MoveXZBlind(RobGoal):

    def __init__(self, rob, pos, dist_thresh):
        super().__init__(rob)
        self.update_target(pos)
        self.aPos = None
        self.dist_thresh = dist_thresh

    def update(self):
        aPos = self.rob.cached['getAgentPos'][0]
        if aPos is not None:
            self.aPos = aPos

    def act(self):
        if self.aPos is None:
            return []
        deltas = self.rob.getYawDeltas(False)
        proj_x = deltas[0] * (self.target[0] - self.aPos[0]) + deltas[2] * (self.target[2] - self.aPos[2])
        proj_y = deltas[0] * (self.target[2] - self.aPos[2]) - deltas[2] * (self.target[0] - self.aPos[0])
        proj_l = math.hypot(proj_x, proj_y)
        if proj_l > 1:
            proj_x /= proj_l
            proj_y /= proj_l
        else:
            proj_x *= 0.33
            proj_y *= 0.33
        return [['strafe', str(proj_y)], ['move', str(proj_x)]]

    def stop(self):
        return [['move', '0'], ['strafe', '0']]

    def finished(self):
        if self.aPos is None:
            return None
        return math.hypot(self.aPos[0] - self.target[0], self.aPos[2] - self.target[2]) < self.dist_thresh

    def update_target(self, pos):
        self.target = pos


class MoveBlockCenter(MoveXZBlind):

    def __init__(self, rob):
        super().__init__(rob, None, 0.05)

    def update(self):
        super().update()
        if self.aPos is not None:
            pos = int_coords(self.aPos[:3])
            self.update_target([pos[0]+0.5, pos[1], pos[2]+0.5])


class DigUnder(RobGoal):

    def __init__(self, rob):
        super().__init__(rob,
            SAnd([
                MoveBlockCenter(rob),
                LookPitch(rob, 3.14/2),
                AttackBlockTool(rob)]))


class MoveAndDirectBlind(RobGoal):

    def __init__(self, rob, pos, dist_thresh):
        super().__init__(rob,
            CAnd([LookAt(rob, pos),
                  MoveXZBlind(rob, pos, dist_thresh)]))

    def update_target(self, pos):
        self.delegate.subgoals[0].update_target(pos)
        self.delegate.subgoals[1].update_target(pos)


class ActT(Goal):

    def __init__(self, a, s, tm, once=False):
        super().__init__()
        self.tm = tm
        self.t0 = None
        self.a = [a]
        self.s = [s]
        self.once = once

    def act(self):
        if self.t0 is None:
            self.t0 = time()
        if self.once:
            a = self.a.copy()
            self.a = []
            self.once = False
            return a
        return [] if self.finished() else self.a

    def stop(self):
        return self.s

    def finished(self):
        return (self.t0 is not None) and (time() - self.t0 > self.tm)


class JumpUpOrObtain(RobGoal):

    def __init__(self, agent):
        rob = agent.rob
        build_blocks = ['dirt', 'grass', 'stone', 'sand', 'sandstone', 'gravel']
        invent = rob.cached['getInventory'][0]
        for block in build_blocks:
            item = rob.minelogy_instance.findInInventory(invent, {'type': block})
            if item:
                super().__init__(rob,
                    SAnd([ActT(['swapInventoryItems', '0', str(item['index'])], [], 0.2, True),
                          LookPitch(rob, 3.14/2),
                          MoveBlockCenter(rob),
                          CAnd([ActT(['jump', '1'], ['jump', '0'], 1.0),
                                ActT(['use', '1'], ['use', '0'], 0.5)])]))
                return
        super().__init__(agent, Obtain(agent, [{'type': 'dirt', 'quantity': 10}]))


class ApproachPos(Switcher):

    def __init__(self, agent, pos, dist_thresh=0.9):
        self.agent = agent
        super().__init__(agent.rob)
        self.target = pos
        self.move = MoveAndDirectBlind(agent.rob, pos, dist_thresh)
        self.dist_thresh = dist_thresh
        self.current_state = ['start']
        self.last_state = None

    def update(self):
        self.last_state = self.current_state
        ga = GridAnalyzer(self.rob, self.target, self.dist_thresh)
        self.current_state = ga.analyzePaths()
        # print(self.last_state, self.current_state)
        strafe = None
        if self.current_state[-1] == 'left':
            strafe = '-1'
        if self.current_state[-1] == 'right':
            strafe = '1'
        if self.delegate is None:
            if self.current_state[0] == 'clean' or self.current_state[0] == 'down':
                self.delegate = self.move
            elif self.current_state[0] == 'swim':
                # move down if diving or if target is below?
                self.delegate = COr([self.move, ActT(['jump', '1'], ['jump', '0'], 1)])
            elif self.current_state[0] == 'mine':
                self.delegate = LookAndAttackBlock(self.rob, self.current_state[2])
                strafe = None
            elif self.current_state[0] == 'jump':
                self.delegate = COr([self.move, ActT(['jump', '1'], ['jump', '0'], 1)])
            elif self.current_state[0] == 'dig' or self.current_state[0] == 'cliff':
                self.delegate = DigUnder(self.rob)
                strafe = None
            elif self.current_state[0] == 'fly':
                self.delegate = JumpUpOrObtain(self.agent)
                strafe = None
            else:
                return #TODO
            if strafe is not None:
                self.delegate = COr([self.delegate, ActT(['strafe', strafe], ['strafe', '0'], 1)])
        else:
            self.stopDelegate = self.stopDelegate or self.current_state[0] != self.last_state[0] or \
                (strafe is not None and self.current_state[-1] != self.last_state[-1])
        super().update()

    def finished(self):
        return self.current_state[0] == 'done'

    def update_target(self, pos):
        self.target = pos
        # it's ok to update target during other actions (e.g. mining)
        self.move.update_target(pos)


class BasicSearch(RobGoal):

    def __init__(self, agent, blocks):
        super().__init__(agent.rob)
        self.delegate = ApproachPos(agent, self.__get_target_pos(0, 0))
        self.blocks = blocks
        self.t0 = time()

    def __get_target_pos(self, dPitch, dYaw, dist=100):
        pos = self.rob.cached['getAgentPos'][0]
        dPos = MCConnector.yawDelta(degree2rad(pos[4]) + dYaw)
        dPos[1] = -math.sin(degree2rad(pos[3]) + dPitch)
        pos = [pos[0], pos[1] + 1.66, pos[2]]
        return [c + dc * dist for c, dc in zip(pos, dPos)]

    def update(self):
        t = time()
        if t - self.t0 > 2:
            dPitch, dYaw = self.next_dir()
            self.delegate.update_target(self.__get_target_pos(dPitch, dYaw))
            self.t0 = t
        super().update()

    def next_dir(self):
        self.set_speed(0.15) # random scan should be not too fast
        aPos = self.rob.cached['getAgentPos'][0]
        return random()-0.5-degree2rad(aPos[3]), random()-0.5

    def set_speed(self, speed):
        self.delegate.move.delegate.subgoals[0].set_speed(speed)

    def finished(self):
        los = self.rob.cached['getLineOfSights'][0]
        if los is None or 'type' not in los:
            return False
        return los['type'] in self.blocks


class NoticeSearch(BasicSearch):

    def __init__(self, agent, blocks):
        super().__init__(agent, blocks)
        self.agent = agent
        self.agent.blockMem.add_focus_blocks(blocks)

    def finished(self):
        # TODO: blocks in ignore_blocks are missed even if they are searched; need to add focus_blocks
        # if self.blockMem.nearestBlock(block) is not None
        return self.agent.blockMem.recallNearest(self.blocks) is not None

    def stop(self):
        self.agent.blockMem.del_focus_blocks(self.blocks)
        return super().stop()


class NeuralSearch(NoticeSearch):

    # TODO: it would be better to get this info rom NeuralWrapper -> RobustObserver
    rec_blocks = ['log', 'leaves', 'coal_ore']

    def next_dir(self):
        self.set_speed(0.6) # random scan should be not too fast
        # check if NeuralSearch 
        if all([(block not in NeuralSearch.rec_blocks) for block in self.blocks]):
            return super().next_dir()
        segm_data = self.rob.getCachedObserve('getNeuralSegmentation')
        if segm_data is not None:
            heatmaps, img = segm_data
            h, w = heatmaps.shape[-2:]
            size = (h // 10, w // 10)
            stride = (3, 3)
            pooled = torch.nn.functional.avg_pool2d(heatmaps[:, 1:], kernel_size=size, stride=stride)
            for idx in range(len(NeuralSearch.rec_blocks)):
                if NeuralSearch.rec_blocks[idx] in self.blocks:
                    heatmap = pooled[0, idx]
                    if heatmap.max() > 0.05:
                        logging.debug('see %s', NeuralSearch.rec_blocks[idx])
                        pix = torch.argmax(heatmap)
                        wid = pooled.size()[-1] 
                        hei = pooled.size()[-2]
                        h_idx = pix.item() // wid
                        w_idx = pix.item() % wid
                        return (h_idx - hei/2) / wid * 1.9, (w_idx - wid/2) / wid * 1.9
        else:
            logging.warn('segm_data is None')
        return super().next_dir()


class PickNear(Switcher):

    def __init__(self, agent, items, max_cnt=None):
        super().__init__(agent.rob)
        self.agent = agent
        self.items = items
        self.cnt = max_cnt if max_cnt is not None else \
            (5 if items[0] == '*' else 20)

    def update(self):
        nearEnt = self.rob.cached['getNearEntities'][0]
        target = None
        # TODO: choose the nearest item first?
        # TODO: reuse rob.nearestFromEntities ?
        for entity in nearEnt:
            # TODO? add items ignored by *?
            if entity['name'] in self.items or \
               ('life' not in entity and self.items[0] == '*'):
                   target = entity
                   break
        if self.delegate is None and target is not None and self.cnt > 0:
            self.cnt -= 1
            # TODO? better finish criterium (entity is picked?)
            self.delegate = ApproachPos(self.agent, [target['x'], target['y'], target['z']], 0.5)
        if self.delegate is not None and target is None:
            self.stopDelegate = True
        super().update()


class SelectMineTool(RobGoal):

    def __init__(self, rob):
        super().__init__(rob)
        self.tool_idx = 0
        self.last_target = None

    def update(self):
        los = self.rob.cached['getLineOfSights'][0]
        inv = self.rob.cached['getInventory'][0]
        if los is not None and 'type' in los and los['type'] != self.last_target and inv is not None:
            mine_entry = self.rob.minelogy_instance.find_mine_by_block({'type': los['type']})
            tool = self.rob.minelogy_instance.select_minetool(inv, mine_entry)
            self.tool_idx = 0 if tool is None else tool['index']
            self.last_target = los['type']
            # TODO: if tool is None and mine_entry is not None:

    def act(self):
        if self.finished():
            return []
        # TODO: if self.tool_idx != 0 +++ hotbar once?
        a = ['swapInventoryItems', '0', str(self.tool_idx)]
        self.tool_idx = 0
        return [a, ['hotbar.1', '1'], ['hotbar.1', '0']]

    def stop(self):
        return [['hotbar.1', '0']]

    def finished(self):
        return self.tool_idx == 0


class FindAndMine(Switcher):

    def __init__(self, agent, blocks, depthmin):
        self.agent = agent
        self.blocks = blocks
        self.stage = 0
        self.last_targ = None
        self.aPos = self.agent.rob.cached['getAgentPos'][0]
        if depthmin is None:
            self.depthmin = self.aPos[1]
        else:
            self.depthmin = depthmin
        # we need to update focus blocks also within ApproachPos and AttackBlockTool
        self.agent.blockMem.add_focus_blocks(blocks)
        super().__init__(agent.rob)

    def update(self):
        self.aPos = self.agent.rob.cached['getAgentPos'][0]
        # we cannot use SAnd, because we don't know `target` for ApproachPos a priori
        # we also need to be able to choose another target during Search and Approach
        # this logic could be somehow unified...
        targ, targ_block = self.agent.nearestBlock(self.blocks, True)
        if self.delegate is None:
            if targ is None and self.stage == 0:
                self.last_targ = None
                self.delegate = NeuralSearch(self.agent, self.blocks)
            elif targ_block != self.blocks[0] and self.depthmin < self.aPos[1] and self.stage == 0:
                self.stage = 0.5
                self.delegate = SAnd([
                    ApproachPos(self.agent, [self.aPos[0], self.aPos[1] - 1, self.aPos[2]], 2.5)
                ])
            elif targ_block != self.blocks[0] and self.depthmin < self.aPos[1] and self.stage == 0.5:
                self.stage = 2
                self.delegate = SAnd([
                    LookAt(self.rob, [self.aPos[0], self.aPos[1] - 1, self.aPos[2]]),
                    AttackBlockTool(self.rob)
                ])
            elif self.stage == 0:
                self.stage = 1
                self.last_targ = targ
                targ = list(map(lambda x: x+0.5, self.last_targ))
                self.delegate = ApproachPos(self.agent, targ, 2.5)
            elif self.stage == 1:
                self.stage = 2
                self.delegate = SAnd([
                    LookAt(self.rob, list(map(lambda x: x+0.5, self.last_targ))),
                    AttackBlockTool(self.rob)
                    , PickNear(self.agent, ['*'])
                    ])
        elif targ is not None and self.last_targ != targ and self.stage < 2:
            self.bRestart = True
            self.stopDelegate = True
            self.last_targ = targ
        super().update()

    def stop(self):
        self.agent.blockMem.del_focus_blocks(self.blocks)
        return super().stop()


class Obtain(Switcher):

    def __init__(self, agent, items):
        super().__init__(agent.rob)
        self.agent = agent
        self.items = items

    def update(self):
        invent = self.rob.waitNotNoneObserve('getInventory', False)
        new_items = list(filter(lambda item: not self.rob.minelogy_instance.isInInventory(invent, item), self.items))
        if self.delegate is not None:
            # TODO TODO: the problem here is that the agent can decide to find log,
            # but encounter log2, start mining it and doesn't reconsider its plan
            # (although it can make plank, it will not)
            # A non-hacky solution is to introduce SOr, which selects one branch
            # to consider, although update other branches checking if they become more
            # achievable (also, it would be nice to associate costs with plans) -- TODO
            self.stopDelegate = new_items != self.items
        if self.delegate is None:
            for item in new_items:
                name = self.rob.minelogy_instance.get_otype(item)
                target = self.rob.nearestFromEntities(name)
                if target is not None:
                    # self.delegate = ApproachPos(self.agent, target)
                    self.delegate = PickNear(self.agent, [name]) #TODO: add radius?
                    break
        if self.delegate is None:
            # we could consider lacking items in parallel, but it's difficult,
            # so we just try to choose the best craft
            best_craft = None
            best_lack = None
            for item in new_items:
                for craft_entry in self.rob.minelogy_instance.find_crafts_by_result(item):
                    lack_items = self.rob.minelogy_instance.lackCraftItems(invent, craft_entry)
                    if best_craft is None or len(best_lack) > len(lack_items):
                        best_craft = craft_entry
                        best_lack = lack_items
            if best_lack is not None:
                if len(best_lack) == 0:
                    t = self.rob.minelogy_instance.get_otype(best_craft[1])
                    i = self.rob.minelogy_instance.findInInventory(invent, best_craft[0][0])
                    t = self.rob.minelogy_instance.checkCraftType(t,i)
                    # t = i['variant'] + ' ' + t
                    t = self.rob.minelogy_instance.addFuel(t, invent)
                    self.delegate = ActT(['craft', t], [], 0.5, True)
                else:
                    self.delegate = Obtain(self.agent, best_lack)
        if self.delegate is None:
            best_goal = None
            best_diff = 10
            for item in new_items:
                for mine_entry in self.rob.minelogy_instance.find_mines_by_result(item):
                    tool = mine_entry[0]['tools'][-1]
                    blocks = [self.rob.minelogy_instance.get_target_variants(b) for b in mine_entry[0]['blocks']]
                    depthmin = None
                    if 'depthmin' in blocks[0]:
                        depthmin = blocks[0]['depthmin']
                    if isinstance(blocks[0], list):
                        blocks = [b['type'] for b in blocks[0]]
                    else:
                        blocks = [b['type'] for b in blocks]
                    if tool is None or self.rob.minelogy_instance.isInInventory(invent, {'type': tool}):
                        # self.chooseTool(invent, tool)
                        if self.agent.nearestBlock(blocks) is not None:
                            diff = 0
                            goal = FindAndMine(self.agent, blocks, depthmin)
                        else:
                            blocks2 = self.rob.minelogy_instance.assoc_blocks(blocks)
                            if blocks2 != [] and self.agent.nearestBlock(blocks2) is not None:
                                diff = 1
                            else:
                                diff = 2
                            goal = FindAndMine(self.agent, blocks + blocks2, depthmin)
                    else:
                        diff = 3
                        goal = Obtain(self.agent, [{'type': tool}])
                    if blocks[0] == 'diamond_ore':
                        sticks_in_inv = self.rob.minelogy_instance.findInInventory(invent, {'type': 'stick'})
                        if sticks_in_inv is None:
                            diff = -1
                            goal = Obtain(self.agent, [{'type': 'stick', 'quantity': 20}])
                        elif sticks_in_inv['quantity'] < 5:
                            diff = -1
                            goal = Obtain(self.agent, [{'type': 'stick', 'quantity': 20}])
                    if diff < best_diff:
                        best_diff = diff
                        best_goal = goal
            self.delegate = best_goal
        if self.delegate is None and new_items != []:
            logging.warn("PANIC: don't know how to obtain %s", str(new_items))
            new_items = []
        self.items = new_items
        super().update()

    def finished(self):
        return self.items == []


class GridAnalyzer:

    def __init__(self, rob, target, dist_thresh=1.5): #target=None, yaw=None, pitch=None
        self.dist_thresh = dist_thresh
        self.grid3D = rob.getNearGrid3D(False)
        self.dimX = len(self.grid3D[0][0])
        self.dimZ = len(self.grid3D[0])
        self.dimY = len(self.grid3D)
        # we need this to know coordinates of the grid
        self.pa = rob.getCachedObserve('getAgentPos')
        self.p0 = [math.floor(p) for p in self.pa[0:3]]
        self.target = target
        self.dp = [t - p for t, p in zip(self.target, self.pa)]
        self.dist = math.hypot(self.dp[0], self.dp[2])
        if self.dist > 1e-6:
            self.dp[0] /= self.dist
            self.dp[2] /= self.dist
            self.dp[1] /= math.hypot(self.dist, self.dp[1])

    def inWater(self):
        b = self.grid3D[self.dimY//2][self.dimZ//2][self.dimX//2]
        return b == 'water' or b == 'flowing_water'

    def underWater(self):
        b = self.grid3D[self.dimY//2+1][self.dimZ//2][self.dimX//2]
        return b == 'water' or b == 'flowing_water'

    def analyzeGridPos(self, pos, level):
        # flat analysis
        MAX_CNT = 100
        for t in range(MAX_CNT):
            xf = pos[0] + t * self.dp[0] * 0.1
            zf = pos[2] + t * self.dp[2] * 0.1
            xc = math.floor(xf)
            zc = math.floor(zf)
            x = xc - self.p0[0] + self.dimX // 2
            z = zc - self.p0[2] + self.dimZ // 2
            if x < 0 or x >= self.dimX or z < 0 or z >= self.dimZ:
                return {'d': MAX_CNT, 'status': 'free'}
            block = self.grid3D[self.dimY//2+level][z][x]
            if block in RobustObserver.deadlyBlocks:
                return {'d': t, 'status': 'deadly'}
            if block not in RobustObserver.passableBlocks and level >= 0 or \
               block in RobustObserver.passableBlocks and block != 'water' and level < 0:
                return {'d': t, 'status': 'obstacle', 'o': [xf, pos[1]+level+0.5, zf]}
        return {'d': MAX_CNT, 'status': 'clean'}
        # + return obstacle?

    def analyzeLine(self, pos, level):
        dx = 0.25 * self.dp[2]
        dz = 0.25 * self.dp[0]
        res = self.analyzeGridPos(pos, level)
        res_sides = [
            self.analyzeGridPos([pos[0] + dx, pos[1], pos[2] - dz], level),
            self.analyzeGridPos([pos[0] - dx, pos[1], pos[2] + dz], level)]
        for r in res_sides:
            if (level >= 0 and r['d'] < res['d']) or (level < 0 and r['d'] > res['d']):
                res = r
        return res

    def analyzePath(self, pos):
        # logging.debug(f"target {self.target},\tpa {self.pa}\tdist {self.dist}\tdp {self.dp}\n")
        # print(f"target {self.target},\tpa {self.pa}\tdist {self.dist}\tdp {self.dp}\n")
        DIST_CL = 9
        self.last_r2 = 0
        dy = self.target[1] - self.pa[1]
        rd = math.hypot(self.dist, dy-1.66)
        if rd < self.dist_thresh or (self.dist < self.dist_thresh and dy < 2.66 and dy > -0.99):
            return ['done', 0]
        if self.target[1] - self.pa[1] > 1.5+self.dist_thresh and self.dist < self.dist_thresh:
            if self.grid3D[self.dimY-1][self.dimZ//2][self.dimX//2] in RobustObserver.passableBlocks:
                #TODO: and not self.inWater()
                return ['fly', 0]
            else:
                return ['mine', 0, [self.pa[0], self.pa[1]+2.5, self.pa[2]]]
        if dy < -0.5 and self.dist < 0.99:
            return ['dig', 0]
        res = [self.analyzeLine(pos, level-2) for level in range(5)]
        mt = min(res[2]['d'], res[3]['d'])
        md = min(self.dist * 10, DIST_CL)
        self.last_r2 = res[2]['d']
        # TODO check if in water; where to check? : just to avoid obstacles or to float? or outside?
        if self.underWater():
            return ['swim', 0]
        if mt > md: # clean path ahead, analyze ground
            ut = res[1]['d']
            if ut > md:
                # if self.target[1] - self.pa[1] < -2: ...ladder
                return ['swim' if self.inWater() else 'clean', min(mt, ut)]
            if any([res[l]['status'] == 'deadly' for l in range(5)]):
                return ['deadly', ut]
            if ut < 6 and res[0]['d'] < 6:
                return ['cliff', ut]
            return ['down', ut]
        else: # obstacles
            # TODO TODO: the logic can be made more complex:
            #   - if dy<-self.dist than mine ladders, else ok to jump or even 'fly'
            # TODO: what if not 'obstacle'???
            if res[3]['d'] <= DIST_CL and res[3]['status'] == 'obstacle':
                return ['mine', res[3]['d'], res[3]['o']]
            if self.target[1] - self.pa[1] >= -1 or self.inWater():
                # prefer to jump
                if res[3]['d'] >= DIST_CL and res[4]['d'] >= DIST_CL:
                    return ['jump', 1]
                if res[4]['d'] <= DIST_CL and res[4]['status'] == 'obstacle':
                    return ['mine', res[4]['d'], res[4]['o']]
                return ['ERROR', res]
            else:
                if res[2]['d'] <= DIST_CL and res[2]['status'] == 'obstacle':
                    return ['mine', res[2]['d'], res[2]['o']]
        return ['ERROR', res]

    def analyzePaths(self):
        res = self.analyzePath(self.pa)
        for s in range(4):
            dx = (1.5 - s) * self.dp[2] / 1.5
            dz = (s - 1.5) * self.dp[0] / 1.5
            r = self.analyzePath([self.pa[0] + dx, self.pa[1], self.pa[2] + dz])
            if self.last_r2 < 3: # avoid considering strafing inside blocks (should be improved?)
                continue
            if r[0] == res[0]:
                continue
            if self.target[1] < self.pa[1] and r[0] == 'down':
                res = r + ['left' if s <= 1 else 'right']
                continue
            if r[0] == 'clean' or (res[0] != 'clean' and r[0] == 'jump') or \
               (r[0] != 'deadly' and res[0] == 'deadly'):
                res = r + ['left' if s <= 1 else 'right']
        return res


class ListenAndDo(Switcher):

    def __init__(self, agent):
        super().__init__(agent.rob)
        self.agent = agent
        self.next_goal = None
        self.terminate = False

    def update(self):
        command = self.rob.getCachedObserve('getChat')
        if self.next_goal is not None:
            self.delegate = self.next_goal
            self.next_goal = None
        elif command is not None:
            words = command[0].split(' ')
            if words[-1] == 'terminate':
                self.terminate = True
            if len(words) > 1:
                if words[-2] == 'get':
                    self.next_goal = Obtain(self.agent, [{'type': words[-1]}])
            if self.next_goal is not None:
                print("Received command: ", command)
                if self.delegate is not None:
                    self.stopDelegate = True
        super().update()

    def finished(self):
        return self.terminate


