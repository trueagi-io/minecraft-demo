import os
import logging
import torch
import numpy
import cv2


model_cache = dict()

def process_pixel_data(pixels, keep_aspect_ratio, maximum_size):
    height, width, _ = pixels.shape
    aspect_ratio_org = width / height
    if maximum_size is not None and maximum_size:
        height = maximum_size[1] if height > maximum_size[1] else height
        if keep_aspect_ratio:
            width = int(height * aspect_ratio_org)
        else:
            width = maximum_size[0] if width > maximum_size[0] else width

    # to make width and height divisible by 8
    width -= width % 8
    height -= height % 8

    img_data = cv2.resize(pixels, dsize=(width, height), interpolation=cv2.INTER_CUBIC)

    return img_data


def get_image(img_frame, keep_aspect_ratio, maximum_size):
    if img_frame is not None:
        return process_pixel_data(img_frame.pixels, keep_aspect_ratio, maximum_size)
    return None


class NeuralWrapper:
    def __init__(self, rob, keep_aspect_ratio=True, maximum_size=(384, 240)):
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

