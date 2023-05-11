from collections import deque
import threading
import cv2

#Old visualizer which uses threading. Not working on Mac. I've left it just in case.

# class Visualizer(threading.Thread):
#     def __init__(self):
#         super().__init__(name='visualization', daemon=False)
#         self.queue = deque(maxlen=10)
#         self._lock = threading.Lock()
#         self._stop = False
#
#     def __call__(self, *args):
#         with self._lock:
#             self.queue.append(args)
#
#     def stop(self):
#         self._stop = True
#
#     def run(self):
#         while not self._stop:
#             while self.queue:
#                 with self._lock:
#                     data = self.queue.pop()
#                 cv2.imshow(data[0], data[1])
#             cv2.waitKey(300)

class Visualizer():
    def __init__(self):
        self.queue = deque(maxlen=10)

    def __call__(self, *args):
        self.queue.append(args)

    def run(self):
        while self.queue:
            data = self.queue.pop()
            cv2.imshow(data[0], data[1])
        cv2.waitKey(1)
