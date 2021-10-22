import cv2
import os
import math
import numpy
from collections import deque, defaultdict
import time
import tagilmo.utils.mission_builder as mb
from tagilmo.utils.malmo_wrapper import MalmoConnector, RobustObserver
from tagilmo.utils import segment_mapping


IMAGE = 0
SEGMENT = 1
BLOCKS = 2


def get_types(segm) -> set:
    """
    Get unique elements from the tensor as a set
    """
    return set(numpy.unique(segm[:,:,0]))


class DataCollection:
    def __init__(self, maxlen, datadir):
        self.queue = deque(maxlen=10)
        # block -> img count 
        self.img_with_block = defaultdict(int)
        self.img_pairs = []
        self.maxlen = maxlen
        self.idx = 0
        self.frame_count = 0
        self.datadir = datadir
        self.load()

    def load(self):
        idx = 0
        for idx in range(len(os.listdir(self.datadir)) // 2): 
            img_path = os.path.join(self.datadir, 'img' + str(idx) + '.png')
            segm_path = os.path.join(self.datadir, 'seg' + str(idx) + '.png')
            if os.path.exists(segm_path):
                segm = cv2.imread(segm_path)
                blocks = get_types(segm)
                self.update_stats(blocks, set())
                self.img_pairs.append((img_path, segm_path, blocks))
        print('loaded {0} files'.format(idx + 1))

    def put(self, img, segm):
        if self.frame_count == 4: 
            blocks = get_types(segm)
            self.queue.append((img, segm, blocks))
            self.idx = self.iteration(self.idx)
            self.frame_count = 0
        else:
            self.frame_count += 1

    def compute_weight(self, blocks):
        result = - 1
        for e in blocks:
            count = max(self.img_with_block.get(e, 1), 1)
            weight = math.log(len(self.img_pairs) / count)
            if result < weight:
                result = weight
        return weight

    def run(self):
        idx = 0
        while True:
            idx = self.iteration(idx)

    def store_item(self, item, idx):
        img_path = os.path.join(self.datadir, 'img' + str(idx) + '.png')
        segm_path = os.path.join(self.datadir, 'seg' + str(idx) + '.png')
        cv2.imwrite(img_path, item[IMAGE])
        cv2.imwrite(segm_path, item[SEGMENT])
        return img_path, segm_path

    def iteration(self, idx):
        if self.queue:
            item = self.queue.pop()
        else:
            return idx
        blocks = item[BLOCKS]
        if len(self.img_pairs) < self.maxlen:
            img_path, segm_path = self.store_item(item, idx)
            self.img_pairs.append((img_path, segm_path, blocks))
            self.update_stats(blocks, set())
            print('new block ', len(self.img_pairs))
        else:
            weight = self.compute_weight(blocks)
            blocks_old = self.img_pairs[idx][BLOCKS]
            weight_current = self.compute_weight(blocks_old)
            if weight_current < weight:
                print('replace new {0} old {1}'.format(weight, weight_current))
                img_path, segm_path = self.store_item(item, idx)
                self.img_pairs[idx] = (img_path, segm_path, blocks)
                self.update_stats(blocks, blocks_old)
        idx += 1
        if self.maxlen <= idx:
            idx = 0
        return idx
    
    def update_stats(self, blocks_new, blocks_old):
        for e in blocks_old:
            if e in self.img_with_block:
                self.img_with_block[e] -= 1

        for e in blocks_new:
            self.img_with_block[e] += 1


def start_mission():
    miss = mb.MissionXML()
    colourmap_producer = mb.ColourMapProducer(width=320 * 4, height=240 * 4)
    video_producer = mb.VideoProducer(width=320 * 4, height=240 * 4, want_depth=False)    
    
    obs = mb.Observations()
    agent_handlers = mb.AgentHandlers(observations=obs)

    agent_handlers = mb.AgentHandlers(observations=obs, 
        colourmap_producer=colourmap_producer,
        video_producer=video_producer)

    miss = mb.MissionXML(agentSections=[mb.AgentSection(name='Cristina',
             agenthandlers=agent_handlers,)])

    world = mb.defaultworld(
        seed='43',
        forceReset="false")

    miss.setWorld(world)
    miss.serverSection.initial_conditions.allowedmobs = "Pig Sheep Cow Chicken Ozelot Rabbit Villager"
    # uncomment to disable passage of time:
    miss.serverSection.initial_conditions.time_pass = 'false'
    miss.serverSection.initial_conditions.time_start = "1000"

    mc = MalmoConnector(miss)
    obs1 = RobustObserver(mc)
    return mc, obs1



def get_img(mc):
    img_data = mc.waitNotNoneObserve('getImage')
    if img_data is not None:
        img_data = img_data.reshape((240 * 4, 320 * 4, 3))
        return img_data


def get_segment(mc):
    img_data = mc.waitNotNoneObserve('getSegmentation')
    if img_data is not None:
        img_data = img_data.reshape((240 * 4, 320 * 4, 3))
        return img_data


def get_distance(mc):
    aPos = mc.getAgentPos()
    coords1 = aPos[:3]
    coords = [mc.getLineOfSight('x'),
              mc.getLineOfSight('y'),
              mc.getLineOfSight('z')]

    height = 1.6025
    coords1[1] += height

    dist = numpy.linalg.norm(numpy.asarray(coords) - numpy.asarray(coords1), 2) 
    return dist


def main():
    mc, obs = start_mission()
    mc.safeStart()
    dataset = DataCollection(400, 'image_data1')
    while True:
        obs.clear()
        img = get_img(obs)
        segm = get_segment(obs)
        visible = mc.getFullStat('LineOfSight')
        if visible:
            dist = get_distance(mc)
            if dist < 3:
                continue
        print(visible)
        if img is not None and segm is not None:
            dataset.put(img, segm)
            cv2.imshow('segm', segm[:, :, 0])
            cv2.imshow('img', img)
            cv2.waitKey()

main()
