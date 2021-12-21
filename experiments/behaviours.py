import logging
import math
import numpy
import torch
import time
from tagilmo.utils.mathutils import normAngle, degree2rad
from b_tree import *


logger = logging.getLogger()

YAW = 4
PITCH = 3


class Base:
    def __init__(self, rob):
       self.rob = rob 

    def __call__(self):
        return self.act()


class CheckVisible(Base):
    def __init__(self, rob, visualizer=None):
        super().__init__(rob)
        self.found_item = None
        self.visualizer = visualizer

    def act(self):
        result = []
        status = 'running'
        segm_data = self.rob.getCachedObserve('getNeuralSegmentation')
        if segm_data is None:
            return status, result
        ray = self.rob.getCachedObserve('getLineOfSights')
        if ray is not None and ray['type'] == 'stone':
            self.found_item = ray 
            return 'success', ['pitch 0', 'turn 0']
        if segm_data is not None:
            heatmaps, img = segm_data
            h, w = heatmaps.shape[-2:]
            size = (h // 10, h // 10)
            stone = heatmaps[0, 4].unsqueeze(0).unsqueeze(0)
            # smooth heatmap a bit
            pooled = torch.nn.functional.avg_pool2d(stone, kernel_size=size, stride=size)
            if self.visualizer is not None:
                self.visualizer('avg_stone', (pooled * 255).long().numpy().astype(numpy.uint8)[0, 0])
                self.visualizer('stone', (stone * 255).long().numpy().astype(numpy.uint8)[0, 0])
            if pooled.max() < 0.3:
                # can't see stone
                return 'failure', []
            idx = torch.argmax(pooled)
            h_pooled, w_pooled = pooled.shape[-2:]
            h_idx = idx // w_pooled 
            w_idx = idx % w_pooled
            pitch = (h_idx.item() - h_pooled / 2) / 80
            turn = (w_idx.item() - w_pooled / 2) / 80 
            result = ['pitch {0}'.format(pitch), 'turn {0}'.format(turn)]

        return status, result 



class TurnTo(Base):
    """
    Align orientation with given pitch and yaw
    """

    def __init__(self, rob, target_pitch=None, target_yaw=None):
        """
        Parameters are in degrees
        """
        super().__init__(rob)
        self.target_pitch = target_pitch if target_pitch is None else degree2rad(target_pitch)
        self.target_yaw = target_yaw if target_yaw is None else degree2rad(target_yaw)

    def act(self):
        result = []
        pitch = 0
        logger.debug('stabilizing')
        pos = self.rob.waitNotNoneObserve('getAgentPos')
        pitch = self.compute_pitch(pos)
        turn = self.compute_turn(pos)
        result.append('pitch {0}'.format(pitch))
        result.append('turn {0}'.format(turn))
        status = 'running'
        if pitch == 0 and turn == 0:
            status = 'success'
        return status, result

    def compute_turn(self, pos):
        if self.target_yaw is None:
            return 0

        current_yaw = degree2rad(pos[YAW])
        diff = - normAngle(current_yaw - self.target_yaw)
        if abs(diff) < 0.05:
            diff = 0
        yaw = diff / 5 
        return yaw

    def compute_pitch(self, pos):
        if self.target_pitch is None:
            return 0
        current_pitch = degree2rad(pos[PITCH])
        # proportional control
        # should work good enough, no need for PID
        diff = - normAngle(current_pitch - self.target_pitch)
        if abs(diff) < 0.05:
            diff = 0
        pitch = diff / 5
        return pitch


class Rotate(Base):
    """rotate 1 rad then return success"""

    def __init__(self, rob):
        self.start_yaw = None
        self.rob = rob

    def act(self):
        result = [] 
        pos = self.rob.waitNotNoneObserve('getAgentPos')
        current_yaw = normAngle(degree2rad(pos[YAW]))
        if self.start_yaw is None:
           self.start_yaw = current_yaw      
        if abs(normAngle(current_yaw - self.start_yaw)) > 1:
            result.append('turn 0')
            return 'success', result
        result.append('turn 0.02')
        return 'running', result


class MountainScan:
    def __init__(self, rob):
        self.rob = rob
        self.root = self._create_tree() 
        self._finished = False
        self.start_yaw = None 

    def _create_tree(self):
        rob = self.rob
        stabilize_rotate = And([TurnTo(rob, -6), Rotate(rob)])
        return Or([CheckVisible(rob), stabilize_rotate])

    def getAngle(self):
        pos = self.rob.waitNotNoneObserve('getAgentPos')
        return normAngle(degree2rad(pos[YAW]))
        
    def act(self):
        assert not self._finished
        if self.start_yaw is None:
            self.start_yaw = self.getAngle()
        status, action = self.root()
        if status ==  'failure': 
             self._finished = True
        if status == 'success':
            # we can either succed by finding a mountain
            # or it just rotate finished
            last_node = self.root.getLastNode()
            if isinstance(last_node, Rotate):
                diff = abs(normAngle(self.start_yaw - self.getAngle()))
                if diff  > 1.7 * math.pi:
                    # we're done
                    self._finished = True
                else:
                    self.root = self._create_tree()
            elif isinstance(last_node, CheckVisible):
                print('found mountain')     
                print(last_node.found_item)
                self._finished = True
            else:
                print('unexpected {0}'.format(last_node))
        return action 

    def finished(self):
        return self._finished

    def precond(self):
        return True
