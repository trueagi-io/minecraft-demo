from collections import deque
import threading
import cv2


class Visualizer(threading.Thread):
    def __init__(self):
        super().__init__(name='visualization', daemon=False)
        self.queue = deque(maxlen=10)
        self._lock = threading.Lock()
        self._stop = False

    def __call__(self, *args):
        with self._lock:
            self.queue.append(args)

    def stop(self):
        self._stop = True

    def run(self):
        while not self._stop:
            while self.queue:
                with self._lock:
                    data = self.queue.pop()
                    if len(data[1].shape) == 3:
                        image = cv2.cvtColor(data[1], cv2.COLOR_BGR2RGB)
                    else:
                        image = data[1]
                cv2.imshow(data[0], image)
            cv2.waitKey(300)

