import os
import logging
import torch
import numpy
import cv2

model_cache = dict()

def get_image(rob, resize, scale, img_name='getImageFrame'):
    img_frame = rob.getCachedObserve(img_name)
    img_data = None
    if img_frame is not None:
        img_data = numpy.frombuffer(img_frame.pixels, dtype=numpy.uint8)
        img_data = img_data.reshape((240 * scale, 320 * scale, 3))
        if resize != 1:
            height, width, _ = img_data.shape
            img_data = cv2.resize(img_data, (int(width * resize), int(height * resize)),
                fx=resize, fy=resize, interpolation=cv2.INTER_NEAREST)

    return img_data


class NeuralWrapper:
    def __init__(self, rob, resize, scale):
        self.net = self.load_model()
        self.rob = rob
        self.resize = resize
        self.scale = scale

    def _get_image(self):
        img_data = get_image(self.rob, self.resize, self.scale)
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
        path = 'experiments/goodpoint.pt'
        logging.info('loading model from %s', path)
        if path in model_cache:
            return model_cache[path]
        from experiments.goodpoint import GoodPoint
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        n_classes = 5 # other, log, leaves, coal_ore, stone
        depth = False
        net = GoodPoint(8, n_classes, n_channels=3, depth=depth, batchnorm=False).to(device)
        if os.path.exists(path):
            model_weights = torch.load(path, map_location=device)['model']
            net.load_state_dict(model_weights)
        model_cache[path] = net
        return net

