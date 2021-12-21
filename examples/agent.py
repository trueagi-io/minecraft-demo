import numpy

from tagilmo.utils.malmo_wrapper import MalmoConnector, RobustObserverWithCallbacks
from neural import NeuralWrapper
from tagilmo.utils.mathutils import *

class TAgent:

    def __init__(self, miss, visualizer=None):
        mc = MalmoConnector(miss)
        mc.safeStart()
        self.rob = RobustObserverWithCallbacks(mc)
        w = miss.agentSections[0].agenthandlers.video_producer.width
        callback = NeuralWrapper(self.rob, 320/w, w//320)
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
