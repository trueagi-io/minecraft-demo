from collections import deque
import threading
import cv2
import matplotlib.pyplot as plt

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
        tmp_dict = {}
        cnt = -1
        while not self._stop:
            while self.queue:
                with self._lock:
                    data = self.queue.pop()
                    if len(data[1].shape) == 3:
                        image = cv2.cvtColor(data[1], cv2.COLOR_BGR2RGB)
                    else:
                        image = data[1]
                    if data[0] not in tmp_dict:
                        cnt += 1
                        tmp_dict.update({data[0]:cnt})
                    plt.figure(tmp_dict[data[0]])
                    plt.clf()
                    plt.imshow(image)
                    plt.title(data[0])
                    # plt.axis("off")
                    # plt.close()
            plt.pause(0.3)
            # cv2.waitKey(300)


