import math
import cv2
import torch
import numpy
import os
from time import sleep, time
from tagilmo.utils.malmo_wrapper import MalmoConnector, RobustObserver
import tagilmo.utils.mission_builder as mb
from examples import minelogy
import logging


def normAngle(angle):
    while (angle < -math.pi): angle += 2 * math.pi
    while (angle > math.pi): angle -= 2 * math.pi
    return angle

def int_coord(x):
    return int(x) if x >= 0 else int(x-0.999)

def int_coords(xs):
    return list(map(int_coord, xs))


class NoticeBlocks:

    def __init__(self):
        self.blocks = {}
        self.max_len = 5
        self.ignore_blocks = ['air', 'grass', 'tallgrass', 'double_plant', 'dirt', 'stone']
        self.dx = 4

    def updateBlock(self, block, pos):
        if block not in self.blocks:
            self.blocks[block] = []
        ps = self.blocks[block]
        for p in ps:
            if abs(p[0] - pos[0]) <= self.dx and \
               abs(p[1] - pos[1]) <= self.dx and \
               abs(p[2] - pos[2]) <= self.dx:
                   return
        ps.append(pos)
        self.blocks[block] = ps[1:] if len(ps) > self.max_len else ps

    def removeIfMissing(self, current_block, blocks, pos):
        for block in blocks:
            if block not in self.blocks or block == current_block:
                continue
            self.blocks[block].remove(pos)

    def updateBlocks(self, rob, focus_blocks=[]):
        grid = rob.cached['getNearGrid'][0]
        sight = rob.cached['getLineOfSights'][0]
        if sight is not None and (sight['type'] not in self.ignore_blocks or sight['type'] in focus_blocks):
            self.updateBlock(sight['type'], int_coords([sight['x'], sight['y'], sight['z']]))
        for i in range(len(grid)):
            bUpdate = grid[i] not in self.ignore_blocks or grid[i] in focus_blocks
            if bUpdate or focus_blocks:
                pos = rob.gridIndexToAbsPos(i, observeReq=False)
                pos = int_coords(pos)
            if focus_blocks:
                self.removeIfMissing(grid[i], focus_blocks, pos)
            if bUpdate:
                self.updateBlock(grid[i], pos)


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
        #aPos = rob.waitNotNoneObserve('getAgentPos')
        aPos = self.rob.cached['getAgentPos'][0]
        if self.pitch is not None:
            dPitch = normAngle(self.pitch - aPos[3] * math.pi / 180.)
            self.fin = self.fin and abs(dPitch) < 0.02
            acts += [["pitch", str(dPitch * 0.4)]]
        if self.yaw is not None:
            dYaw = normAngle(self.yaw - aPos[4] * math.pi / 180.)
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
        self.target_pos = pos
        pitch, yaw = self.rob.dirToAgentPos(self.target_pos, observeReq=True)
        self.lookDir = LookDir(rob, pitch, yaw)

    def precond(self):
        return True

    def act(self):
        pitch, yaw = self.rob.dirToAgentPos(self.target_pos, observeReq=False)
        self.lookDir.update_target(pitch, yaw)
        return self.lookDir.act()

    def finished(self):
        return self.lookDir.finished()

    def stop(self):
        return self.lookDir.stop()


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


class ApproachXZPos:

    def __init__(self, rob, pos):
        self.rob = rob
        self.target_pos = pos
        self.move = ForwardNJump(rob)
        self.lookAt = LookAt(rob, pos)

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
        if los is not None:
            if los['inRange']:
                acts = [['attack 1']]
            else:
                acts = [['attack 0']]
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
        self.net = self.load_model()
        # for debug purposes
        self._visualize = True

    def load_model(self):
        path = 'experiments/goodpoint.pt'
        if path in model_cache:
            return model_cache[path]
        from experiments.goodpoint import GoodPoint
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        n_classes = 4 # other, log, leaves, coal_ore
        depth = False
        net = GoodPoint(8, n_classes, n_channels=3, depth=depth, batchnorm=False).to(device)
        if os.path.exists(path):
            model_weights = torch.load(path, map_location=device)['model']
            net.load_state_dict(model_weights)
        model_cache[path] = net
        return net

    def _get_image(self):
        img_frame = self.rob.waitNotNoneObserve('getImageFrame')
        img_data = None
        if img_frame is not None:
            img_data = numpy.frombuffer(img_frame.pixels, dtype=numpy.uint8)
            img_data = img_data.reshape((240, 320, 3))
            img_data = torch.as_tensor(img_data).permute(2,0,1)
            img_data = img_data.unsqueeze(0) / 255.0
        return img_data

    def visualize(self, img, heatmaps):
        if self._visualize:
            cv2.imshow('image', (img * 255).long().numpy().astype(numpy.uint8)[0].transpose(1,2,0))
            cv2.imshow('leaves', (heatmaps[0, 2].detach().numpy() * 255).astype(numpy.uint8))
            cv2.imshow('log', (heatmaps[0, 1].detach().numpy() * 255).astype(numpy.uint8))
            cv2.imshow('coal_ore', (heatmaps[0, 3].detach().numpy() * 255).astype(numpy.uint8))
            cv2.waitKey(200)

    def act(self):
        LOG = 1
        LEAVES = 2
        img = self._get_image()
        turn = 0
        pitch = 0
        if img is not None:
            with torch.no_grad():
                heatmaps = self.net(img)
            h, w = heatmaps.shape[-2:]
            size = (h // 10, w // 10)
            pooled = torch.nn.functional.avg_pool2d(heatmaps[:, 1:], kernel_size=size, stride=size)
            stabilize = True
            log = pooled[0, 0]
            leaves = pooled[0, 1]
            coal_ore = pooled[0, 2]
            blocks = {'log': log, 'leaves': leaves, 'coal_ore': coal_ore}
            for block in blocks.keys():
                self.visualize(img, heatmaps)
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
                pitch = pos[3]
                if pitch < -10:
                    pitch = 0.03
                if 10 < pitch:
                    pitch = - 0.03
        else:
            logging.warn('img is None')
        result = [["turn", str(turn)], ["pitch", str(pitch)]]
        logging.debug(result)
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
        return None if (los is None or los['type'] is None or not los['inRange']) else los['distance']

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


class TAgent:

    def __init__(self, miss):
        mc = MalmoConnector(miss)
        mc.safeStart()
        self.rob = RobustObserver(mc)
        self.blockMem = NoticeBlocks()
        #Not necessary now
        #sleep(2)
        #self.rob.sendCommand("jump 1")
        #sleep(2)
        #self.rob.sendCommand("jump 0")
        #sleep(0.1)

    def howtoMine(self, targ):
        t = minelogy.get_otype(targ[0]) # TODO?: other blocks?
        if t == 'log':
            target = targ + [{'type': 'log2'}, {'type': 'leaves'}, {'type': 'leaves2'}]
        elif t == 'stone':
            target = targ + [{'type': 'dirt'}, {'type': 'grass'}]
        else:
            target = targ
        for targ in target:
            t = minelogy.get_otype(targ)
            ray = self.rob.cached['getLineOfSights'][0]
            if minelogy.matchEntity(ray, targ):
                if ray['inRange']:
                    return [['mine', [ray]]]
                return [['mine', [ray]], ['approach', ray]]
            known = self.rob.nearestFromGrid(t, observeReq=False)
            if known is None:
                if t in self.blockMem.blocks:
                    # TODO? updateBlocks
                    known = self.blockMem.blocks[t][-1]
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

        invent = self.rob.cached['getInventory'][0]
        nearEnt = self.rob.cached['getNearEntities'][0]

        acts = []

        for item in invent:
            if not minelogy.matchEntity(item, target):
                continue
            if 'quantity' in target:
                if item['quantity'] < target['quantity']:
                    continue
            return acts + [['tool' if tool else 'inventory', item]]

        for ent in nearEnt:
            if not minelogy.matchEntity(ent, target):
                continue
            return acts + [['approach', ent]]

        # TODO actions can be kept hierarchically, or we can somehow else
        # analyze/represent that some actions can be done in parallel
        # (e.g. mining of different blocks which don't require unavailable tools)
        # while others cannot be done right away
        # (craft requiring mining, mining requiring tools)
        # TODO? combining similar actions with summing up amounts
        # (may not be necessary with the above)

        for craft in minelogy.crafts:
            if not minelogy.matchEntity(craft[1], target):
                continue
            acts += [['craft', target]]
            for ingrid in craft[0]:
                # TODO? amounts
                act = self.howtoGet(ingrid, craft_only)
                acts = None if act is None else acts + act
            return acts

        if craft_only:
            return None

        for mine in minelogy.mines:
            if not minelogy.matchEntity(mine[1], target):
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
        self.rob.updateAllObservations()
        self.blockMem.updateBlocks(self.rob)
        skill = self.skill
        if skill.precond() and not skill.finished():
            acts = skill.act()
            for act in acts:
                self.rob.sendCommand(' '.join(act))
            return True
        else:
            acts = skill.stop()
            for act in acts:
                self.rob.sendCommand(' '.join(act))
            return False
        return True

    def loop(self):
        self.skill = None
        target = {'type': 'wooden_pickaxe'}
        while target:
            sleep(0.2)
            self.rob.updateAllObservations()
            howto = self.howtoGet(target)
            if howto == []:
                target = None
                break
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
                break
            if howto[-1][0] == 'search':
                self.skill = NeuralSearch(self.rob, self.blockMem, minelogy.get_otlist(howto[-1][1]))
            if howto[-1][0] == 'craft':
                t = minelogy.get_otype(howto[-1][1])
                if t == 'planks': # hotfix
                    invent = self.rob.cached['getInventory'][0]
                    for item in invent:
                        if item['type'] == 'log':
                            t = item['variant'] + ' ' + t
                            break
                self.rob.craft(t)
                continue
            if howto[-1][0] == 'approach':
                self.skill = ApproachXZPos(self.rob,
                                [howto[-1][1]['x'], howto[-1][1]['y'], howto[-1][1]['z']])
            if howto[-1][0] == 'mine':
                #self.skill = MineAround(self.rob, minelogy.get_otlist(howto[-1][1]))
                self.skill = MineAtSight(self.rob)
            if self.skill is None:
                break
            while self.ccycle():
                sleep(0.05)
            self.skill = None


if __name__ == '__main__':
    video_producer = mb.VideoProducer(width=320, height=240, want_depth=False)
    agent_handlers = mb.AgentHandlers(video_producer=video_producer)
    miss = mb.MissionXML(agentSections=[mb.AgentSection(name='Cristina',
             agenthandlers=agent_handlers,)])


    world = mb.flatworld("3;7,25*1,3*3,2;1;stronghold,biome_1,village,decoration,dungeon,lake,mineshaft,lava_lake", seed='43', forceReset="false")
    miss.serverSection.initial_conditions.time_pass = 'false'
    miss.serverSection.initial_conditions.time_start = "1000"
    world1 = mb.defaultworld(forceReset="false")
    miss.setWorld(world)
    miss.serverSection.initial_conditions.allowedmobs = "Pig Sheep Cow Chicken Ozelot Rabbit Villager"
    agent = TAgent(miss)
    agent.loop()
