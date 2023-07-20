import torch
import random
import numpy
from time import sleep, time
import tagilmo.utils.mission_builder as mb
from tagilmo.utils.vereya_wrapper import MCConnector
from examples.minelogy import Minelogy
from mcdemoaux.agenttools.agent import TAgent
import logging
from examples.log import setup_logger
from mcdemoaux.vision.vis import Visualizer

from tagilmo.utils.mathutils import *

from examples.knowledge_lists import *

class MoveForward:

    def __init__(self, rob):
        self.rob = rob
        self.aPos = rob.waitNotNoneObserve('getAgentPos')
        self.isAct = False

    def precond(self):
        path = self.rob.analyzeGridInYaw(observeReq=False)
        return path['safe'] and path['passWay'] and path['solid']

    # def expect(self):

    def act(self):
        self.isAct = True
        return [['move', '1']]

    def stop(self):
        self.isAct = False
        return [['move', '0']]

    def finished(self):
        return False


class Jump:

    def __init__(self):
        self.isAct = False

    def precond(self):
        return True # actually, should check the grid above

    def act(self):
        self.isAct = True
        return [['jump', '1']]

    def stop(self):
        self.isAct = False
        return [['jump', '0']]

    def finished(self):
        return False


class ForwardNJump:

    def __init__(self, rob):
        self.rob = rob
        self.move = MoveForward(rob)
        self.jump = Jump()
        self.fin = False
        # dist = ?

    def precond(self):
        # path = self.rob.analyzeGridInYaw(observeReq=False)
        # return path['level'] < 2 and path['safe']
        return True

    def act(self):
        acts = []
        if self.fin:
            return acts
        path = self.rob.analyzeGridInYaw(observeReq=False)
        grid = self.rob.getNearGrid3D(observeReq=False)
        slc = grid[len(grid)//2]
        slc = slc[len(slc)//2]
        block = slc[len(slc)//2]
        if block == 'water' or path['level'] == 1:
            acts += self.jump.act()
        elif self.jump.isAct:
            acts += self.jump.stop()
        acts += self.move.act()
        if path['level'] > 1:
            acts += [['attack', '1']]
        return acts

    def stop(self):
        self.fin = True
        return self.move.stop() + self.jump.stop()

    def finished(self):
        return self.fin


class LookDir:

    def __init__(self, rob, pitch, yaw):
        self.fin = False
        self.update_target(pitch, yaw)
        self.rob = rob

    def precond(self):
        return True

    def act(self):
        acts = []
        self.fin = True
        aPos = self.rob.cached['getAgentPos'][0]
        if self.pitch is not None:
            dPitch = normAngle(self.pitch - degree2rad(aPos[3]))
            self.fin = self.fin and abs(dPitch) < 0.02
            acts += [["pitch", str(dPitch * 0.4)]]
        if self.yaw is not None:
            dYaw = normAngle(self.yaw - degree2rad(aPos[4]))
            self.fin = self.fin and abs(dYaw) < 0.02
            acts += [["turn", str(dYaw * 0.4)]]
        return acts

    def finished(self):
        return self.fin

    def stop(self):
        return [["turn", "0"], ["pitch", "0"]]

    def update_target(self, pitch, yaw):
        self.pitch = pitch
        self.yaw = yaw


class LookAt:

    def __init__(self, rob, pos):
        self.rob = rob
        self.lookDir = LookDir(rob, 0, 0)
        self.update_target(pos, True)

    def precond(self):
        return True

    def act(self):
        self.update_target()
        return self.lookDir.act()

    def finished(self):
        return self.lookDir.finished()

    def stop(self):
        return self.lookDir.stop()

    def update_target(self, pos=None, observeReq=False):
        if pos is not None:
            self.target_pos = pos
        pitch, yaw = self.rob.dirToAgentPos(self.target_pos, observeReq)
        self.lookDir.update_target(pitch, yaw)


class VisScan:

    def __init__(self):
        self.t0 = time()

    def precond(self):
        return True

    def act(self):
        t = time()
        turnVel = 0.1 * math.cos((t-self.t0) * 1.3)
        pitchVel = -0.025 * math.cos((t-self.t0) * 6)
        return [["turn", str(turnVel)], ["pitch", str(pitchVel)]]

    def stop(self):
        return [["turn", "0"], ["pitch", "0"]]


class StatePredictor:

    def __init__(self, rob):
        self.rob = rob
        self.stuck_thresh = 0.005

    def precond(self):
        return True

    def is_stucked(self):
        curr_pos = numpy.array(self.rob.cached['getAgentPos'][0])
        prev_pos = numpy.array(self.rob.cached_buffer['getAgentPos'][0])
        if prev_pos is None or curr_pos is None or prev_pos.shape == ():
            return False
        action_magnitude = numpy.linalg.norm(curr_pos-prev_pos)
        if action_magnitude < self.stuck_thresh and action_magnitude > 1e-6:
            return True
        else:
            return False


class Perturbation:
    def __init__(self, rob):
        self.rob = rob
        sigma_trans = 5
        self.stuck_pred = StatePredictor(rob)
        self.move = ForwardNJump(rob)
        self.pos = self.rob.cached['getAgentPos'][0]
        new_x = random.gauss(self.pos[0], sigma_trans)
        new_y = self.pos[1]
        new_z = random.gauss(self.pos[2], sigma_trans)
        new_yaw = 0
        new_pitch = 0
        self.target_pos = [new_x, new_y, new_z, new_yaw, new_pitch]
        self.lookAt = LookAt(rob, self.target_pos)

    def precond(self):
        return self.stuck_pred.is_stucked()

    def act(self):
        aPos = self.rob.cached['getAgentPos'][0]
        pos = self.target_pos
        if abs(aPos[0] - pos[0]) < 1.4 and abs(aPos[2] - pos[2]) < 1.4 and not self.move.finished():
            return self.move.stop()
        los = self.rob.cached['getLineOfSights'][0]
        acts = []
        if los is not None:
            if los['inRange']:
                acts = [['attack', '1']]
            else:
                acts = [['attack', '0']]
        return self.move.act() + self.lookAt.act() + acts


class ApproachXZPos:

    def __init__(self, rob, pos):
        self.rob = rob
        self.target_pos = pos
        self.move = ForwardNJump(rob)
        self.lookAt = LookAt(rob, pos)
        self.perturb = Perturbation(rob)

    def precond(self):
        # return self.move.precond()
        return True

    def act(self):
        aPos = self.rob.cached['getAgentPos'][0]
        pos = self.target_pos
        if abs(aPos[0] - pos[0]) < 1.4 and abs(aPos[2] - pos[2]) < 1.4 and not self.move.finished():
            return self.move.stop()
        los = self.rob.cached['getLineOfSights'][0]
        acts = []
        if los is not None and los['hitType'] != 'MISS':
            if los['inRange']:
                acts = [['attack', '1']]
            else:
                acts = [['attack', '0']]
        if self.perturb.precond():
            acts += self.perturb.act()
            print("Got stucked. Need perturbation.")
        return self.move.act() + self.lookAt.act() + acts

    def stop(self):
        return self.move.stop() + self.lookAt.stop()

    def finished(self):
        return self.move.finished() and self.lookAt.finished()


model_cache = dict()


class NeuralScan:
    def __init__(self, rob, blocks):
        self.rob = rob
        self.blocks = blocks

    def act(self):
        self.rob.observeProcCached()
        logging.debug("scanning for {0}".format(self.blocks))
        LOG = 1
        LEAVES = 2
        turn = 0
        pitch = 0
        segm_data = self.rob.getCachedObserve('getNeuralSegmentation')
        if segm_data is not None:
            heatmaps, img = segm_data
            h, w = heatmaps.shape[-2:]
            size = (h // 10, w // 10)
            pooled = torch.nn.functional.avg_pool2d(heatmaps[:, 1:], kernel_size=size, stride=size)
            stabilize = True
            log = pooled[0, 0]
            leaves = pooled[0, 1]
            coal_ore = pooled[0, 2]
            blocks = {'log': log, 'leaves': leaves, 'coal_ore': coal_ore}
            for block in blocks.keys():
                if block in self.blocks:
                    m = blocks[block].max()
                    if 0.1 < m:
                        logging.debug('see %s', block)
                        idx = torch.argmax(blocks[block])
                        h_idx = idx // 10
                        w_idx = idx % 10
                        pitch = (h_idx.item() - 4) / 80
                        turn = (w_idx.item() - 4) / 60
                        stabilize = False
                        break
            if stabilize:
                logging.debug('stabilizing')
                pos = self.rob.waitNotNoneObserve('getAgentPos')
                current_pitch = pos[3]
                if current_pitch < -10:
                    pitch = 0.03
                if 10 < current_pitch:
                    pitch = - 0.03
        else:
            logging.warn('img is None')
        result = [["turn", str(turn)], ["pitch", str(pitch)]]
        self.rob.observeProcCached()
        return result

    def stop(self):
        return [["turn", "0"], ["pitch", "0"]]


class NeuralSearch:
    def __init__(self, rob, blockMem, blocks):
        self.blocks = blocks
        self.blockMem = blockMem
        self.move = ForwardNJump(rob)
        self.scan = NeuralScan(rob, blocks)

    def precond(self):
        return self.move.precond()

    def act(self):
        return self.move.act() + self.scan.act()

    def stop(self):
        return self.move.stop() + self.scan.stop()

    def finished(self):
        for block in self.blocks:
            # TODO: blocks in ignore_blocks are missed even if they are searched; need to add focus_blocks
            # if self.blockMem.nearestBlock(block) is not None
            if block in self.blockMem.blocks and len(self.blockMem.blocks[block]) > 0:
                return True
        return False


class Search4Blocks:

    def __init__(self, rob, blockMem, blocks):
        self.blocks = blocks
        self.blockMem = blockMem
        # blockMem.addToFocus + removeFromFocus
        self.move = ForwardNJump(rob)
        self.scan = VisScan()

    def precond(self):
        return self.move.precond()

    def act(self):
        return self.move.act() + self.scan.act()

    def stop(self):
        return self.move.stop() + self.scan.stop()

    def finished(self):
        for block in self.blocks:
            # TODO: blocks in ignore_blocks are missed even if they are searched; need to add focus_blocks
            # if self.blockMem.nearestBlock(block) is not None
            if block in self.blockMem.blocks and len(self.blockMem.blocks[block]) > 0:
                return True
        return False


class MineAtSight:

    def __init__(self, rob):
        self.rob = rob
        self.isAct = False
        self.dist = self._get_dist()

    def _get_dist(self):
        los = self.rob.cached['getLineOfSights'][0]
        return None if (los is None or los['hitType'] == 'MISS' or los['type'] is None or not los['inRange']) else los['distance']

    def precond(self):
        return self.dist is not None

    def act(self):
        return [['attack', '1']]

    def stop(self):
        return [['attack', '0']]

    def finished(self):
        dist = self._get_dist()
        if self.dist is None or dist is None:
            return True
        return abs(dist - self.dist) > 0.1


class MineAround:

    def __init__(self, rob, objs):
        self.rob = rob
        self.objs = objs
        self.mine = None
        self.approach = None
        self._set_target()

    def _set_target(self):
        for obj in self.objs:
            self.target = self.rob.nearestFromGrid(obj, observeReq=False)
            if self.target is not None:
                self.approach = ApproachXZPos(self.rob, self.target)
                break

    def precond(self):
        return self.target is not None

    def act(self):
        acts = []
        if self.mine is not None:
            if self.mine.finished():
                acts += self.mine.stop()
                self.mine = None
                self._set_target()
            else:
                return acts + self.mine.act()
        if self.approach is not None:
            if self.approach.finished():
                acts += self.approach.stop()
                self.approach = None
                self.mine = MineAtSight(self.rob)
            else:
                return acts + self.approach.act()
        return acts

    def stop(self):
        acts = []
        if self.approach is not None:
            acts += self.approach.stop()
        if self.mine is not None:
            acts += self.mine.stop()
        return acts

    def finished(self):
        return not self.precond()


class LJAgent(TAgent):
    def __init__(self, mc, visualizer=None):
        super().__init__(mc, visualizer)
        self.mlogy = None

    def set_mlogy(self, mlogy):
        self.mlogy = mlogy

    def howtoMine(self, targ):
        target = targ + self.mlogy.get_oatargets(targ[0]) # TODO?: other blocks?
        # if t == 'log':
        #     target = targ + [{'type': 'log2'}, {'type': 'leaves'}, {'type': 'leaves2'}]
        # if t == 'stone':
        #     target = targ + [{'type': 'dirt'}, {'type': 'grass'}]
        # elif t == 'coal_ore' or t == 'iron_ore':
        #     target = targ + [{'type': 'stone'}]
        # else:
        #     target = targ
        for targ in target:
            t = self.mlogy.get_otype(targ)
            t_list = [var['type'] for var in self.mlogy.get_target_variants(targ, True)]
            ray = self.rob.getCachedObserve('getLineOfSights')
            if (ray['hitType'] != 'MISS'):
                if self.mlogy.matchEntity(ray, targ):
                    if ray['inRange']:
                        return [['mine', [ray]]]
                    return [['mine', [ray]], ['approach', ray]]
            known = self.rob.nearestFromGrid(t_list, observeReq=False)
            if known is None:
                for _t in t_list:
                    if _t in self.blockMem.blocks:
                        # TODO? updateBlocks
                        known = self.blockMem.blocks[_t][-1]
                        break
            # REM: 'approach' should include lookAt
            if known is not None:
                return [['mine', [targ]], ['approach', {'type': t, 'x': known[0], 'y': known[1], 'z': known[2]}]]
        return [['mine', target], ['search', target]]

    def howtoGet(self, target, craft_only=False, tool=False):
        '''
        This method doesn't try to return a complete plan.
        For example, if there is a matching nearby entity, it will
        proposes to approach it without planning to mine remaining quantity
        '''

        if target is None or not isinstance(target, dict):
            return []

        invent = self.rob.cached['getInventory'][0]
        nearEnt = self.rob.cached['getNearEntities'][0]

        acts = []

        for item in invent:
            if not self.mlogy.matchEntity(item, target):
                continue
            if 'quantity' in target:
                if item['quantity'] < target['quantity']:
                    continue
            return acts + [['tool' if tool else 'inventory', item]]

        for ent in nearEnt:
            if not self.mlogy.matchEntity(ent, target):
                continue
            return acts + [['approach', ent]]

        # TODO actions can be kept hierarchically, or we can somehow else
        # analyze/represent that some actions can be done in parallel
        # (e.g. mining of different blocks which don't require unavailable tools)
        # while others cannot be done right away
        # (craft requiring mining, mining requiring tools)
        # TODO? combining similar actions with summing up amounts
        # (may not be necessary with the above)

        # There can be multiple ways to craft something (e.g. planks from log or log2)
        best_way = None
        for craft in self.mlogy.crafts:
            if not self.mlogy.matchEntity(craft[1], target):
                continue
            next_acts = [['craft', target]]
            for ingrid in craft[0]:
                # TODO? amounts
                act = self.howtoGet(ingrid, craft_only)
                if act is None or 'UNKNOWN' in act:
                    next_acts = None
                    break
                if (act[-1][0] == 'approach') and (len(act) == 3):
                    new_act_type = self.mlogy.checkCraftType(act[-3][1], act[-1][1])
                    if new_act_type is not None:
                        act[-3][1] = new_act_type
                next_acts += act
            if next_acts is not None:
                if best_way is None or len(best_way) > len(next_acts):
                    best_way = next_acts
        if best_way is not None:
            return acts + best_way

        if craft_only:
            return None

        for mine in self.mlogy.mines:
            if not self.mlogy.matchEntity(mine[1], target):
                continue
            # TODO: there can be alternative blocks to mine (OR instead of AND)
            acts += self.howtoMine(mine[0]['blocks'])
            best_cnd = None
            for tool in mine[0]['tools']:
                if tool is None:
                    best_cnd = []
                    break
                cnd = self.howtoGet({'type': tool}, craft_only=True, tool=True)
                if cnd is None:
                    continue
                if len(cnd) <= 1:
                    best_cnd = cnd
                    break
                if best_cnd is None or len(cnd) < len(best_cnd):
                    best_cnd = cnd
            if best_cnd is None:
                best_cnd = self.howtoGet({'type': mine[0]['tools'][-1]}, craft_only=False)
            acts += best_cnd
            return acts
        return ['UNKNOWN']

    def ccycle(self):
        self.blockMem.updateBlocks(self.rob)
        skill = self.skill
        if skill.precond() and not skill.finished():
            acts = skill.act()
            logging.debug(acts)
            for act in acts:
                self.rob.sendCommand(act)
            return True
        else:
            acts = skill.stop()
            logging.debug(acts)
            for act in acts:
                self.rob.sendCommand(act)
            return False
        return True

    def loop(self, target = None):
        self.skill = None
        while target != 'terminate':
            sleep(0.05)
            self.rob.updateAllObservations()
            self.visualize()

            # print("Current state: ", self.rob.cached['getAgentPos'])
            # print("Prev state: ", self.rob.cached_buffer['getAgentPos'])

            # In Minecraft chat:
            # '/say @p get stone_pickaxe'
            # '/say @p stop'
            # '/say @p terminate'
            chat = self.rob.getCachedObserve('getChat')[0]
            if chat is not None and chat[0] is not None:
                print("Receive chat: ", chat[0])
                words = chat[0].split(' ')
                if words[-2] == 'get':
                    target = {'type': words[-1]}
                else:
                    if words[-1] == 'stop':
                        target = None
                        self.skill = None
                        self.rob.sendCommand('move 0')
                        self.rob.sendCommand('jump 0')
                        self.rob.sendCommand('attack 0')
                    elif words[-1] == 'terminate':
                        break
                self.rob.cached['getChat'] = (None, self.rob.cached['getChat'][1])

            if self.skill is not None:
                if self.ccycle():
                    continue
                self.skill = None

            howto = self.howtoGet(target)

            if howto == []:
                target = None
            elif howto[-1][0] == 'UNKNOWN':
                print("Panic. Don't know how to get " + str(target))
                print(str(howto))
                break
            while howto[-1][0] == 'inventory' or howto[-1][0] == 'tool':
                if howto[-1][0] == 'tool' and howto[-1][1]['index'] != 0:
                    self.rob.mc.sendCommand('swapInventoryItems 0 ' + str(howto[-1][1]['index']))
                howto = howto[:-1]
                if howto == []:
                    target = None
                    break
            if target is None or howto == []:
                target = 'terminate'
                continue
            if howto[-1][0] == 'search':
                blocks = [block['type'] for block in self.mlogy.get_target_variants(howto[-1][1][0], True)]
                self.skill = NeuralSearch(self.rob, self.blockMem, blocks)
                continue
            if howto[-1][0] == 'craft':
                t = self.mlogy.get_otype(howto[-1][1])
                invent = self.rob.cached['getInventory'][0]
                t_recipes = self.mlogy.find_crafts_by_result(t)
                for t_recipe in t_recipes:
                    to_mine = t_recipe[0][0]
                    for inv in invent:
                        if self.mlogy.matchEntity(inv, to_mine):
                            new_t = self.mlogy.checkCraftType(howto[-1][1], inv)
                            if new_t is not None:
                                t = self.mlogy.get_otype(new_t)
                                break
                self.rob.craft(t)
                sleep(0.2)
                continue
            if howto[-1][0] == 'approach':
                self.skill = ApproachXZPos(self.rob,
                                           [howto[-1][1]['x'], howto[-1][1]['y'], howto[-1][1]['z']])
                continue
            if howto[-1][0] == 'mine':
                #self.skill = MineAround(self.rob, self.mlogy.get_otlist(howto[-1][1]))
                self.skill = MineAtSight(self.rob)
                continue
            if self.skill is None:
                print("Panic. No skill available for " + str(target))
                print(str(howto))
                break


if __name__ == '__main__':
    setup_logger()
    visualizer = Visualizer()
    visualizer.start()
    agent_handlers = mb.AgentHandlers(video_producer=mb.VideoProducer())
    miss = mb.MissionXML(agentSections=[mb.AgentSection(name='Cristina',
             agenthandlers=agent_handlers,)])

    # world = mb.flatworld("3;7,25*1,3*3,2;1;stronghold,biome_1,village,decoration,dungeon,lake,mineshaft,lava_lake", seed='43', forceReset="false")
    miss.serverSection.initial_conditions.time_pass = 'false'
    miss.serverSection.initial_conditions.time_start = "1000"
    miss.setWorld(mb.defaultworld(forceReset="true"))
    miss.serverSection.initial_conditions.allowedmobs = "Pig Sheep Cow Chicken Ozelot Rabbit Villager"
    agent = LJAgent(MCConnector(miss), visualizer=visualizer)

    # initialize minelogy
    item_list, recipes = agent.rob.getItemsAndRecipesLists()
    blockdrops = agent.rob.getBlocksDropsList()
    mlogy = Minelogy(item_list, items_to_craft, recipes, items_to_mine, blockdrops, ore_depths)
    agent.set_mlogy(mlogy)

    agent.rob.sendCommand("chat /difficulty peaceful")
    # agent.loop()
    agent.loop(target = {'type': 'wooden_pickaxe'})

    visualizer.stop()

