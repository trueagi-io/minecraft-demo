from collections import deque
import threading
import cv2
import platform

class VisualizerThreaded(threading.Thread):
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
                cv2.imshow(data[0], data[1])
            cv2.waitKey(300)

class VisualizerMac():
    def __init__(self):
        self.queue = deque(maxlen=10)

    def __call__(self, *args):
        self.queue.append(args)
        self.run()

    def stop(self):
        pass

    def start(self):
        pass

    def run(self):
        if len(self.queue) >= 8:
            while self.queue:
                data = self.queue.pop()
                cv2.imshow(data[0], data[1])
            cv2.waitKey(20)

class VisualizerBlank():
    def __init__(self):
        pass

    def __call__(self, *args):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def run(self):
        pass

def Visualizer(blankvis=None):
    ostype = platform.system()
    if (blankvis is None and ostype == 'Darwin') or blankvis == True:
        return VisualizerBlank()
    elif blankvis == False and ostype == 'Darwin':
        return VisualizerMac()
    elif (blankvis is None or blankvis == False) and (ostype == 'Linux' or ostype == 'Windows'):
        return VisualizerThreaded()
    else:
        print("unknown OS type, initializing blank Visualizer\n")
        return VisualizerBlank()
