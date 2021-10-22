"""
Train neural network on pairs of rgb and segmented images
"""
import torch
import os.path
import numpy
import cv2
from tagilmo.utils import segment_mapping
from dataset import MinecraftSegmentation


reverse_map = {v: k for k, v in segment_mapping.items()}

# merge second to first
to_merge = ('log/oak', 'log/birch'), ('leaves/oak', 'leaves/birch')
to_train = ['log/oak', 'leaves/oak', 'dirt', 'grass', 'lava', 'water', 'stone/stone']
to_train = ['log/oak', 'leaves/oak']
train_id = [reverse_map[k] for k in to_train]

RESIZE = 1/4


def replace(segm, to_merge):
    """replace second element from to_merge pairs with the first one"""
    for (first, second) in to_merge:
        f_id = reverse_map[first]
        s_id = reverse_map[second]
        diff = f_id - s_id
        segm += (diff * (segm == s_id)).astype('uint8')
    return segm


def transform_item_1channel(item):
    image, segm_image = item
    height, width, _ = image.shape
    segm_image1 = replace(segm_image[:, :, 0].copy(), to_merge)
    mask = numpy.zeros_like(segm_image1)
    for t in to_train:
        mask += (segm_image1 == reverse_map[t])
    segm_image1 *= mask
    return image, segm_image1


def transform_item_nchannel(item):
    image, segm_image = item

    height, width, _ = image.shape
    image = cv2.resize(image, (int(width * RESIZE), int(height * RESIZE)),
                fx=RESIZE, fy=RESIZE, interpolation=cv2.INTER_NEAREST)

    segm_image = cv2.resize(segm_image, (int(width * RESIZE), int(height * RESIZE)),
                fx=RESIZE, fy=RESIZE, interpolation=cv2.INTER_NEAREST)

    height, width, _ = image.shape
    segm_image1 = replace(segm_image[:, :, 0].copy(), to_merge)
    mask = numpy.zeros((height, width, len(to_train) + 1))
    for i, t in enumerate(to_train):
        mask[:, :, i + 1] = (segm_image1 == reverse_map[to_train[i]])
    mask[:, :, 0] = ~ (mask[:, :, 1:].sum(2) > 0)
    img = image.transpose(2, 0, 1) / 255
    return img.astype(numpy.float32), mask.transpose(2, 0, 1)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from dataset import MinecraftImageDataset
    from goodpoint import GoodPoint
    import torch.optim as optim
    device = 'cpu'
    device = 'cuda'
    train_depth = False
    n_epochs = 22
    batch_size = 22

    data_set = MinecraftSegmentation(imagedir='image_data1',
                                     transform=transform_item_nchannel)
    data_set[44]
    loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    # +1 for None
    net = GoodPoint(8, len(to_train) + 1, n_channels=3, depth=train_depth).to(device)
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
    show = True
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
            # don't use weighting for now
            # logprob *= weights.unsqueeze(0).unsqueeze(2).unsqueeze(3).to(logprob)
            loss = (- (logprob * target)).mean()
            loss.backward()
            optimizer.step()
            if j % 10 == 0:
                print(loss)
                cv2.imshow('leaves', (blocks[0][1] * 255).detach().cpu().numpy().astype(numpy.uint8))
                cv2.imshow('target', (target[0][1] * 255).detach().cpu().numpy().astype(numpy.uint8))
                cv2.imshow('image', (imgs[0].permute(1, 2, 0) * 255).detach().cpu().numpy().astype(numpy.uint8))
                cv2.waitKey(1000)
        snap = dict()
        snap['model'] = net.state_dict()
        snap['optimizer'] = optimizer.state_dict()
        torch.save(snap, 'goodpoint.pt')

