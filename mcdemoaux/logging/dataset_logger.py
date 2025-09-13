from mcdemoaux.vision.neural import *
import json
import datetime


class DatasetLogger:

    def __init__(self, pth_to_saved_data = "dataset/"):
        self.g_counter = 0
        self.pth_to_saved_data = pth_to_saved_data
        if not os.path.exists(self.pth_to_saved_data):
            os.makedirs(self.pth_to_saved_data)

    def _get_image(self, img_frame, resize, scale):
        if img_frame is not None:
            return process_pixel_data(img_frame.pixels, resize, None)#scale)
        return None

    def _generate_name(self):
        now = datetime.datetime.now()
        timestr = now.strftime("%Y%m%d-%H%M%S%f")
        self.g_counter += 1
        fname = timestr + '_' + str(self.g_counter)
        return fname

    def _save_files(self, rob, filename, action):
        SCALE = 4
        img = self._get_image(rob.getCachedObserve('getImageFrame'), SCALE, SCALE)
        cv2.imwrite(self.pth_to_saved_data + filename + ".jpg", img)
        a = {filename: action}
        with open(self.pth_to_saved_data + filename, "w") as fp:
            json.dump(a, fp)

    def logImgActData(self, rob, action):
        fname = self._generate_name()
        self._save_files(rob, fname, action)

