import math
from time import sleep, time
from tagilmo.utils.malmo_wrapper import MalmoConnector, RobustObserver
import tagilmo.utils.mission_builder as mb

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
        if abs(aPos[0] - pos[0]) < 2 and abs(aPos[2] - pos[2]) < 2 and not self.move.finished():
            return self.move.stop()
        return self.move.act() + self.lookAt.act()

    def stop(self):
        return self.move.stop() + self.lookAt.stop()

    def finished(self):
        return self.move.finished() and self.lookAt.finished()


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
        target = None
        while target is None:
            self.skill = Search4Blocks(self.rob, self.blockMem, ['log', 'leaves'])
            while self.ccycle():
                sleep(0.05)
            b = self.blockMem.blocks
            if 'leaves' in b and b['leaves']:
                target = b['leaves'][-1]
            if 'log' in b and b['log']:
                target = b['log'][-1]
        self.skill = ApproachXZPos(self.rob, target)
        while self.ccycle():
            sleep(0.05)
        self.skill = MineAround(self.rob, ['log', 'leaves'])
        while self.ccycle():
            sleep(0.05)


if __name__ == '__main__':
    
    miss = mb.MissionXML()
    miss.setWorld(mb.defaultworld(forceReset="true"))
    miss.serverSection.initial_conditions.allowedmobs = "Pig Sheep Cow Chicken Ozelot Rabbit Villager"
    agent = TAgent(miss)
    agent.loop()
