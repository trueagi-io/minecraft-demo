import os
import logging
import torch
import numpy
import cv2


model_cache = dict()


def process_pixel_data(pixels, resize, scale):
    # img_data = numpy.frombuffer(pixels, dtype=numpy.uint8)
    img_data = cv2.resize(pixels, dsize=(320 * scale, 240 * scale), interpolation=cv2.INTER_CUBIC)
    # img_data = numpy.resize(pixels, (240 * scale, 320 * scale, 4))
    # img_data = pixels
    if resize != 1:
        height, width, _ = img_data.shape
        img_data = cv2.resize(img_data, (int(width * resize), int(height * resize)),
            fx=resize, fy=resize, interpolation=cv2.INTER_NEAREST)

    return img_data


def get_image(img_frame, resize, scale):
    if img_frame is not None:
        return process_pixel_data(img_frame.pixels, resize, scale)
    return None


class NeuralWrapper:
    def __init__(self, rob, resize, scale):
        self.net = self.load_model()
        self.rob = rob
        self.resize = resize
        self.scale = scale

    def _get_image(self):
        img_name = 'getImageFrame'
        img_data = get_image(self.rob.getCachedObserve(img_name), self.resize, self.scale)
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

