import os
import numpy
import logging
from tagilmo import VereyaPython
import json
import time
import tagilmo.utils.mission_builder as mb
from tagilmo.utils.vereya_wrapper import MCConnector, RobustObserver
from common import init_mission
import cv2


logger = logging.getLogger(__name__)


def main():
        start = (-112.0, -190.0)
        mc, rob = init_mission(None, start_x=start[0], start_z=start[1], seed='5', forceReset='true')
        assert mc.safeStart()
        time.sleep(4)


        mc.sendCommand('turn 0.1')
        while True:
            img_frame = rob.waitNotNoneObserve('getImageFrame')
            img_frame1 = rob.waitNotNoneObserve('getSegmentationFrame')
            img_depth = rob.waitNotNoneObserve('getDepthFrame')
            cv2.imshow("image", img_frame.pixels)
            cv2.imshow("segmentation", img_frame1.pixels)
            cv2.imshow("depth", norm_image(img_depth.pixels))
            print(rob.getCachedObserve('getLineOfSights'))
            cv2.waitKey(50)


        mc.stop()



 
def norm_image(pixels):
    d = pixels.astype(numpy.float32)
    valid = d > 0
    if numpy.any(valid):
        d_valid = d[valid]
        hi = 12000
        lo = 1
        # lo, hi = numpy.percentile(d_valid, [1, 85])  # 1–85% range
        d = numpy.clip(d, lo, hi)
        d_norm = (d - lo) / (hi - lo + 1e-6)
        d_vis = (1.0 - d_norm) * 255.0
        d_vis = numpy.clip(d_vis, 0, 255).astype(numpy.uint8)
    else:
        d_vis = numpy.zeros_like(d, dtype=numpy.uint8)
    return d_vis


if __name__ == '__main__':
   VereyaPython.setupLogger()
   main()
