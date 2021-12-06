import cv2
import torch
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
WIDTH = 320 * 4
HEIGHT = 240 * 4


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
        self.prev_time = 0
        self.datadir = datadir
        if not os.path.exists(self.datadir):
            os.mkdir(self.datadir)
        self.load()
        self.idx = max(len(self.img_pairs) - 1, 0)

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
        t = time.time()
        if t - self.prev_time > 0.4:
            blocks = get_types(segm)
            self.queue.append((img, segm, blocks))
            self.idx = self.iteration(self.idx)
            self.prev_time = t

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
    colourmap_producer = mb.ColourMapProducer(width=WIDTH, height=HEIGHT)
    video_producer = mb.VideoProducer(width=WIDTH, height=HEIGHT, want_depth=False)

    obs = mb.Observations()
    agent_handlers = mb.AgentHandlers(observations=obs)

    agent_handlers = mb.AgentHandlers(observations=obs,
        colourmap_producer=colourmap_producer,
        video_producer=video_producer)

    miss = mb.MissionXML(agentSections=[mb.AgentSection(name='Cristina',
             agenthandlers=agent_handlers,)])

    # good point seed='2', x=-90, y=71, z=375
    # good point seed='3', x=6, y=71, z=350
    world = mb.defaultworld(
        seed='2',
        forceReset="false")

    world1 = mb.flatworld("3;7,25*1,3*3,2;1;stronghold,biome_1,village,decoration,dungeon,lake,mineshaft,lava_lake",
            seed='43',
            forceReset="false")
    miss.setWorld(world1)
    miss.serverSection.initial_conditions.allowedmobs = "Pig Sheep Cow Chicken Ozelot Rabbit Villager"
    # uncomment to disable passage of time:
    miss.serverSection.initial_conditions.time_pass = 'false'
    miss.serverSection.initial_conditions.time_start = "1000"

    mc = MalmoConnector(miss)
    obs1 = RobustObserver(mc)
    return mc, obs1



def get_img(mc):
    img_frame = mc.waitNotNoneObserve('getImageFrame')
    if img_frame is None:
        return None, None
    pos = numpy.asarray((img_frame.pitch, img_frame.yaw, img_frame.xPos, img_frame.yPos, img_frame.zPos))
    return pos, img_frame


def get_segment(mc):
    img_frame = mc.waitNotNoneObserve('getSegmentationFrame')
    if img_frame is None:
        return None, None
    pos = numpy.asarray((img_frame.pitch, img_frame.yaw, img_frame.xPos, img_frame.yPos, img_frame.zPos))
    return pos, img_frame


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


def extract(data):
    img_data = numpy.frombuffer(data, dtype=numpy.uint8)
    img_data = img_data.reshape((HEIGHT, WIDTH, 3))
    return img_data


def main():
    mc, obs = start_mission()
    mc.safeStart()
    test_model = True
    dataset = DataCollection(400, 'train')
    prev_pos = None
    check_dist = False
    while True:
        obs.clear()
        pos1, img = get_img(obs)
        pos2, segm = get_segment(obs)
        if pos1 is None or pos2 is None:
            continue
        diff = numpy.max(numpy.abs(pos1 - pos2))
        if 0 < diff:
            continue
        if prev_pos is not None:
            if numpy.max(numpy.abs(pos1 - prev_pos)) == 0:
                print('old data')
                continue
        if check_dist:
            visible = mc.getFullStat('LineOfSight')
            if visible:
    #            print(visible)
                dist = get_distance(mc)
                if dist < 3:
                    continue
            else:
                continue
        if img is not None and segm is not None:
            img = extract(img.pixels)
            segm = extract(segm.pixels)
            dataset.put(img, segm)
            cv2.imshow('segm', segm)
            cv2.imshow('img', img)
            cv2.waitKey(200)
            prev_pos = pos1
main()
