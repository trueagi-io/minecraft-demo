import logging
import numpy
import math

from tagilmo.utils.malmo_wrapper import MalmoConnector, RobustObserverWithCallbacks
from examples.neural import NeuralWrapper
from tagilmo.utils.mathutils import *

class TAgent:

    def __init__(self, miss, visualizer=None):
        mc = MalmoConnector(miss)
        mc.safeStart()
        self.rob = RobustObserverWithCallbacks(mc)
        vp = miss.agentSections[0].agenthandlers.video_producer
        if vp is not None:
            callback = NeuralWrapper(self.rob, 320/vp.width, vp.width//320)
                                    # cb_name, on_change event, callback
            self.rob.addCallback('getNeuralSegmentation', 'getImageFrame', callback)
        self.blockMem = NoticeBlocks()
        self.visualizer = visualizer

    def visualize(self):
        if self.visualizer is None:
            return
        segm_data = self.rob.getCachedObserve('getNeuralSegmentation')
        if segm_data is None:
            return
        heatmaps, img = segm_data
        # self.visualizer('image', (img * 255).long().numpy().astype(numpy.uint8)[0].transpose(1,2,0))
        self.visualizer('leaves', (heatmaps[0, 2].cpu().detach().numpy() * 255).astype(numpy.uint8))
        self.visualizer('log', (heatmaps[0, 1].cpu().detach().numpy() * 255).astype(numpy.uint8))
        self.visualizer('coal_ore', (heatmaps[0, 3].cpu().detach().numpy() * 255).astype(numpy.uint8))

    def nearestBlock(self, blocks):
        bc = self.rob.blockCenterFromRay()
        los = self.rob.cached['getLineOfSights'][0]
        if (los['hitType'] != 'MISS'):
            if 'minecraft:' in los['type']:
                los['type'] = los['type'].split("minecraft:")[1]
            if bc is not None and los['inRange'] and los['type'] in blocks:
                return int_coords(bc)
        res = self.rob.nearestFromGrid(blocks, False)
        # print("target block coords: "+str(res)+"\n")
        return int_coords(res) if res is not None \
            else self.blockMem.recallNearest(blocks, self.rob.cached['getAgentPos'][0])


class NoticeBlocks:

    def __init__(self):
        self.blocks = {}
        self.max_len = 5
        self.ignore_blocks = ['air', 'grass', 'tallgrass', 'double_plant', 'dirt', 'stone']
        self.dx = 4
        self.focus_blocks = set()

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
            if pos in self.blocks[block]:
                self.blocks[block].remove(pos)

    def add_focus_blocks(self, blocks):
        self.focus_blocks |= set(blocks)

    def del_focus_blocks(self, blocks):
        self.focus_blocks -= set(blocks)

    def updateBlocks(self, rob):
        grid = rob.cached['getNearGrid'][0]
        if grid is None:
            logging.warning('grid is None')
            return
        sight = rob.cached['getLineOfSights'][0]
        bc = rob.blockCenterFromRay()
        if bc is not None and \
          (sight['type'] not in self.ignore_blocks or sight['type'] in self.focus_blocks):
            self.updateBlock(sight['type'], int_coords(bc))
        for i in range(len(grid)):
            bUpdate = grid[i] not in self.ignore_blocks or grid[i] in self.focus_blocks
            if bUpdate or self.focus_blocks != set():
                pos = rob.gridIndexToAbsPos(i, observeReq=False)
                pos = int_coords(pos)
            if self.focus_blocks != set():
                self.removeIfMissing(grid[i], self.focus_blocks, pos)
            if bUpdate:
                self.updateBlock(grid[i], pos)

    def recallNearest(self, targets, aPos=None):
        if aPos is None: aPos = [0,0,0]
        dist = 1e+16
        res = None
        for b in targets:
            if b in self.blocks:
                for pos in self.blocks[b]:
                    dy = aPos[1] + 0.5 - pos[1]
                    dr = math.hypot(aPos[0] - pos[0], aPos[2] - pos[2])
                    if dr < 1 and dy < 0: dr += 2 # avoid blocks under feet
                    d = dr + abs(dy)*10 # y direction is more difficult
                    if d < dist:
                        dist = d
                        res = pos
        return res
