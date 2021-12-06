"""
Train neural network on pairs of rgb and segmented images
"""
import random
import torch
import os.path
import numpy
import cv2
from tagilmo.utils import segment_mapping
from dataset import MinecraftSegmentation


def random_transformer(x, transformers=[]):
    orig_shape = x.shape
    assert (len(orig_shape) == 3)
    assert (orig_shape[2] <= orig_shape[0])
    assert (orig_shape[2] <= orig_shape[1])

    x = x.transpose(2, 0, 1)
    for transform in transformers:
        assert 'resize' not in str(transform.__class__).lower()
        if random.random() < 0.2:
            x_std = x.std()
            if x_std < 0.01:
                continue
            new_x = transform(x)
            if new_x.std() < 0.1 * x_std:
                print("{0} decreased std to {1} from {2}".format(transform, new_x.std(), x_std))
                continue
            x = new_x
    x = x.astype(numpy.float32)
    x = x.transpose(1, 2, 0)
    return x


class RandomTransformer:
    def __init__(self, transformers):
        self.transformers = transformers

    def __call__(self, x):
        return random_transformer(x, self.transformers)


reverse_map = {v: k for k, v in segment_mapping.items()}

# merge second to first
# to do ('leaves/oak', 'vine')
to_merge = ('log/oak', 'log/birch'), ('log/oak', 'log/spruce'),  \
    ('leaves/oak', 'leaves/birch'), ('leaves/oak', 'leaves/spruce'), \
    ('log/oak', 'log/oak1'), ('log/oak', 'log/birch1'), ('log/oak', 'log/spruce1'), \
    ('leaves/oak', 'leaves2/dark_oak'), \
    ('log/oak', 'log2/dark_oak'), ('log/oak', 'log2/dark_oak1'), \
    ('stone/stone', 'stone/granite'), ('stone/stone', 'stone/diorite'), \
    ('stone/stone', 'stone/cobblestone'), ('stone/stone', 'stone/andesite')

to_train = ['log/oak', 'leaves/oak', 'coal_ore', 'stone/stone']
train_id = [reverse_map[k] for k in to_train]

RESIZE = 1/4


def replace(segm, to_merge):
    """replace second element from to_merge pairs with the first one"""
    for (first, second) in to_merge:
        f_id = numpy.asarray(reverse_map[first], numpy.uint8)
        s_id = numpy.asarray(reverse_map[second], numpy.uint8)
        idx = numpy.where(numpy.all(segm == s_id, axis=2))
        segm[idx] = f_id
    return segm


def transform_item_1channel(item):
    image, segm_image = item
    height, width, _ = image.shape
    segm_image1 = replace(segm_image.copy(), to_merge)
    mask = numpy.zeros_like(segm_image1)
    for t in to_train:
        mask += (segm_image1 == reverse_map[t])
    segm_image1 *= mask
    return image, segm_image1


def make_noisy_transformers():
    from noise import AdditiveGaussian, RandomBrightness, AdditiveShade, MotionBlur, SaltPepper, RandomContrast
    # ColorInversion doesn't seem to be usefull on most datasets
    transformer = [
                   AdditiveGaussian(var=30),
                   RandomBrightness(range=(-50, 50)),
                   AdditiveShade(kernel_size_range=[45, 85],
                                 transparency_range=(-0.25, .45)),
                   SaltPepper(),
                   MotionBlur(max_kernel_size=5),
                   RandomContrast([0.6, 1.05])
                   ]
    return RandomTransformer(transformer)

random_t = make_noisy_transformers()

def transform_item_nchannel(item):
    image, segm_image = item

    height, width, _ = image.shape
    assert height == 240 * 4
    image = cv2.resize(image, (int(width * RESIZE), int(height * RESIZE)),
                fx=RESIZE, fy=RESIZE, interpolation=cv2.INTER_NEAREST)

    segm_image = cv2.resize(segm_image, (int(width * RESIZE), int(height * RESIZE)),
                fx=RESIZE, fy=RESIZE, interpolation=cv2.INTER_NEAREST)

    height, width, _ = image.shape
    segm_image1 = replace(segm_image.copy(), to_merge)
    mask = numpy.zeros((height, width, len(to_train) + 1))
    for i, t in enumerate(to_train):
        mask[:, :, i + 1] = numpy.all(segm_image1 == reverse_map[to_train[i]], axis=2)
    mask[:, :, 0] = ~ (mask[:, :, 1:].sum(2) > 0)

    image1 = random_t(image)
    # cv2.imshow('transformed', image1.astype(numpy.uint8))
    img = image1.transpose(2, 0, 1) / 255
    return img.astype(numpy.float32), mask.transpose(2, 0, 1)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from dataset import MinecraftImageDataset
    from goodpoint import GoodPoint
    import torch.optim as optim

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_depth = False
    train = True
    n_epochs = 100
    batch_size = 22

    data_set = MinecraftSegmentation(imagedir='train',
                                     transform=transform_item_nchannel)
    loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    # +1 for None
    net = GoodPoint(8, len(to_train) + 1, n_channels=3, depth=train_depth, batchnorm=False).to(device)
    model_path = 'goodpoint.pt'
    if os.path.exists(model_path):
        model_weights = torch.load(model_path, map_location=device)['model']
        net.load_checkpoint(model_weights)
    net.to(device)
    net.train()
    lr = 0.003
    epochs = 30
    eps = 0.0000001
    optimizer = optim.AdamW(net.parameters(), lr=lr)
    show = False
    for epoch in range(n_epochs):
        for j, batch in enumerate(loader):
            optimizer.zero_grad()
            imgs, target  = batch
            imgs = imgs.to(device)
            target = target.to(device)
            prediction = net(imgs.to(device))
            if train_depth:
                blocks, p_depth = prediction
            else:
                blocks = prediction
            logprob = torch.log(blocks + eps)
            block_count = target.sum(dim=(0, 2,3))
            block_sum = block_count.sum()
            ratio = block_count / block_sum
            weights = torch.as_tensor([1 if x == 0 else 1 / x for x in ratio])
            # weights = torch.as_tensor([0.1, 1, 1]).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(logprob)
            # don't use weighting for now
            # logprob *= weights.unsqueeze(0).unsqueeze(2).unsqueeze(3).to(logprob)
            loss = 0
            if train:
                loss = (- (logprob * target)).mean()
                loss.backward()
                optimizer.step()
            if j % 10 == 0:
                print(loss)
                if show:
                    for i in range(len(blocks)):
                        cv2.imshow('leaves', (blocks[i][1] * 255).detach().cpu().numpy().astype(numpy.uint8))
                        cv2.imshow('target', (target[i][1] * 255).detach().cpu().numpy().astype(numpy.uint8))
                        cv2.imshow('image', (imgs[i].permute(1, 2, 0) * 255).detach().cpu().numpy().astype(numpy.uint8))
                        cv2.waitKey(1000)
        if train:
            snap = dict()
            snap['model'] = net.state_dict()
            snap['optimizer'] = optimizer.state_dict()
            torch.save(snap, 'goodpoint.pt')

