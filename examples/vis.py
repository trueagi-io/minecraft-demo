from collections import deque
import threading
import cv2


class Visualizer(threading.Thread):
    def __init__(self):
        super().__init__(name='visualization', daemon=False)
        self.queue = deque(maxlen=10)
        self._lock = threading.Lock()

    def __call__(self, *args):
        with self._lock:
            self.queue.append(args)
    
    def run(self):
        while True:
            while self.queue:
                with self._lock:
                    data = self.queue.pop()
                cv2.imshow(*data)
            cv2.waitKey(300)


