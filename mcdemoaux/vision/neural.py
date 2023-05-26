import os
import logging
import torch
import numpy
import cv2


model_cache = dict()


def process_pixel_data(pixels, keep_aspect_ratio=False, maximum_area=None):
    # img_data = numpy.frombuffer(pixels, dtype=numpy.uint8)
    height, width, _ = pixels.shape
    wscale = width / 320
    if not keep_aspect_ratio:
        hscale = height / 240
    else:
        hscale = wscale

    scaled_width = int(320*wscale)
    scaled_height = int(240*hscale)

    #to make width and height divisible by 8
    scaled_width -= scaled_width % 8
    scaled_height -= scaled_height % 8

    img_data = cv2.resize(pixels, dsize=(scaled_width, scaled_height), interpolation=cv2.INTER_CUBIC)
    # img_data = numpy.resize(pixels, (240 * scale, 320 * scale, 4))
    # img_data = pixels
    # if resize != 1:
    #     height, width, _ = img_data.shape
    #     img_data = cv2.resize(img_data, (int(width * resize), int(height * resize)),
    #         fx=resize, fy=resize, interpolation=cv2.INTER_NEAREST)

    return img_data


def get_image(img_frame, keep_aspect_ratio=False, maximum_area=None):
    if img_frame is not None:
        return process_pixel_data(img_frame.pixels, keep_aspect_ratio, maximum_area)
    return None


class NeuralWrapper:
    def __init__(self, rob, keep_aspect_ratio=False, maximum_area=None):
        self.net = self.load_model()
        self.rob = rob
        self.keep_aspect_ratio = keep_aspect_ratio
        self.maximum_area = maximum_area

    def _get_image(self):
        img_name = 'getImageFrame'
        img_data = get_image(self.rob.getCachedObserve(img_name), self.keep_aspect_ratio, self.maximum_area)
        # img_data = self.rob.getCachedObserve(img_name).pixels
        if img_data is not None:
            img_data = torch.as_tensor(img_data).permute(2,0,1)
            img_data = img_data.unsqueeze(0) / 255.0
        return img_data

    def __call__(self):
        img = self._get_image()
        if img is not None:
            with torch.no_grad():
                heatmaps = self.net(img)
                return heatmaps, img

    def load_model(self):
        from mcdemoaux import vision
        pth = os.path.dirname(vision.__file__)
        path = pth+'/goodpoint.pt'
        logging.info('loading model from %s', path)
        if path in model_cache:
            return model_cache[path]
        from mcdemoaux.vision.goodpoint import GoodPoint
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        n_classes = 5 # other, log, leaves, coal_ore, stone
        depth = False
        net = GoodPoint(8, n_classes, n_channels=3, depth=depth, batchnorm=False).to(device)
        if os.path.exists(path):
            model_weights = torch.load(path, map_location=device)['model']
            net.load_state_dict(model_weights)
        model_cache[path] = net
        return net

