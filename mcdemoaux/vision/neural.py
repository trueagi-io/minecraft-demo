import os
import logging
import torch
import numpy
import cv2


model_cache = dict()


def process_pixel_data(pixels, keep_aspect_ratio, maximum_size):
    height, width, _ = pixels.shape
    aspect_ratio_org = width / height
    wscale = width // 320
    scaled_width = int(320*wscale)
    scaled_height = int(int(keep_aspect_ratio) * (scaled_width / aspect_ratio_org) + (1 - int(keep_aspect_ratio)) * int(240*wscale))
    if maximum_size is not None and maximum_size:
        scaled_width = maximum_size[0] if scaled_width > maximum_size[0] else scaled_width
        if not keep_aspect_ratio:
            scaled_height = maximum_size[1] if scaled_height > maximum_size[1] else scaled_height
        else:
            scaled_height = int(scaled_width / aspect_ratio_org)

    # to make width and height divisible by 8
    scaled_width -= scaled_width % 8
    scaled_height -= scaled_height % 8

    img_data = cv2.resize(pixels, dsize=(scaled_width, scaled_height), interpolation=cv2.INTER_CUBIC)

    return img_data


def get_image(img_frame, keep_aspect_ratio, maximum_size):
    if img_frame is not None:
        return process_pixel_data(img_frame.pixels, keep_aspect_ratio, maximum_size)
    return None


class NeuralWrapper:
    def __init__(self, rob, keep_aspect_ratio=False, maximum_size=(320, 240)):
        self.net = self.load_model()
        self.rob = rob
        self.keep_aspect_ratio = keep_aspect_ratio
        self.maximum_size = maximum_size

    def _get_image(self):
        img_name = 'getImageFrame'
        img_data = get_image(self.rob.getCachedObserve(img_name), self.keep_aspect_ratio, self.maximum_size)
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

