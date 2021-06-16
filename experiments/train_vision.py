import pickle
import random
import os
from dataset import MinecraftImageDataset
import torch
import common
from math import sin, cos
import numpy
import cv2


def process_img(img):
    _, height, width = img.shape
    img = img.permute(1,2,0).numpy()
    img = (img * 255).astype(numpy.uint8)
    assert img.shape[2] == 3
    img = numpy.ascontiguousarray(img)
    img = cv2.circle(img, (round(width / 2), round(height / 2)), 2, (0,255,0), -1)
    return img


def project_to_origin(range_i, range_j, data, K):
    show = False
    state = data['origin']
    img = state['image'][0:3]
    _, height, width = img.shape
    assert (_ < height) and (_ < width)
    img1 = process_img(img)
    print('pitch {0}, yaw {1}'.format(*state['pitch_yaw']))
    if show:
        cv2.imshow('origin', img1)
        cv2.waitKey()
    pitch_yaw = state['pitch_yaw']
    points_origin = [(common.visible_block_num[state['visible'][0]], (160, 120))]
    for i in range_i:
        for j in range_j:
            key = (i, j)
            if key in data:
                state = data[key]
                img = state['image'][0:3]
                img = process_img(img)
                print(key)
                diff = pitch_yaw - state['pitch_yaw']
                diff[1] *= -1
                print('pitch {0}, yaw {1}'.format(*state['pitch_yaw']))
                print('dff pitch {0}, yaw {1}'.format(*diff))
                pitch, yaw = diff
                if diff[0] < 0:
                    print('bottom')
                else:
                    print('top')
                if diff[1] < 0:
                    print('left')
                else:
                    print('right')
                R = rotation_matrix(0, pitch, yaw)
                visible = state['visible']
                label = common.visible_block_num[visible[0]]
                # take unit vector, since we are working with
                # center pixel
                vec = [100, 0, 0]
                # rotate vector, then project to pixels by calibration matrix
                vec_new = R @ vec
                vec_new /= vec_new[0]
                # x is depth, change coordinates as expected by calibration matrix
                vec_new1 = vec_new[[1, 2, 0]]
                vec_cam_o = K @ vec_new1
                x = round(vec_cam_o[0])
                z = round(vec_cam_o[1])
                if 0 < x < width and 0 < z < height:
                    if show:
                        img1 = cv2.circle(img1, (x, z), 2, (0,255,255), -1)
                    points_origin.append((label, (x, z)))
                if show:
                    cv2.imshow('origin', img1)
    episode_data = [(state['image'][0:3], points_origin, state['image'][3])]
    return points_origin, episode_data


def origin_to_images(key, data, K, K_inv, points_origin):
    show = False
    episode_data = []

    state = data['origin']
    pitch_yaw = state['pitch_yaw']
    if key in data:
        state = data[key]
        img = state['image'][0:3].clone()
        _, height, width = img.shape
        assert (_ < height) and (_ < width)

        img = process_img(img)

        print(key)
        diff = pitch_yaw - state['pitch_yaw']
        diff[1] *= -1
        pitch, yaw = diff
    else:
        return []
    # project all points origin -> last image
    points_proj = []
    R = rotation_matrix(0, pitch, yaw)
    for (label, (x, z)) in points_origin:
        vec_new_p = K_inv @ [x, z, 1]
        vec_new_p = vec_new_p[[2, 0, 1]]
        vec_proj = R.T @ vec_new_p
        X = K @ vec_proj[[1,2,0]]
        X /= X[2]
        x, z = X[0:2]
        x = round(x)
        z = round(z)
        if 0 < x < width and 0 < z < height:
            points_proj.append((label, (x, z)))
            if show:
                img = cv2.circle(img, (x, z), 2, (0,255,255), -1)
                cv2.imshow('move', img)
    img = state['image']
    episode_data.append((img[0:3], points_proj, img[3]))
    if show:
        cv2.imshow('episode', process_img(episode_data[-1][0]))
        cv2.imshow('episode_depth', numpy.ascontiguousarray(episode_data[-1][2]))
        cv2.imshow('episode_dirt', numpy.ascontiguousarray(episode_data[-1][1][7]))
        cv2.waitKey()
    return episode_data


def load_episode(path):
    K = numpy.asarray([ 1.6740810033016248e+02, 0., 160., 0., 1.6740810033016248e+02,
       120., 0., 0., 1. ]).reshape((3,3))
    K_inv = numpy.linalg.inv(K)
    data = pickle.load(open(path, 'rb'))
    img = data['origin']['image'][0:3]
    _, height, width = img.shape
    # +1 since range is not inclusive
    range_i = range(max([x[0] for x in data.keys() if isinstance(x[0], int)]) + 1)
    range_j = range(max([x[1] for x in data.keys() if isinstance(x[1], int)]) + 1)
    # x, z(x to left, z to top)
    img = data['origin']['image']
    # episode data is (rgb image, label tensor, depth map)
    points_origin, episode_data = project_to_origin(range_i, range_j, data, K)
    for i in range_i:
        for j in range_j:
            episode_data += origin_to_images((i, j), data, K, K_inv, points_origin)
    return episode_data


def rotation_matrix(roll, pitch, yaw):
    yaw_mat = numpy.asarray([[cos(yaw), -sin(yaw), 0],
                              [sin(yaw), cos(yaw), 0],
                              [0, 0, 1]])

    pitch_mat = numpy.asarray([[cos(pitch), 0 , sin(pitch)],
                                [0,     1,  0],
                                [-sin(pitch), 0, cos(pitch)]])

    roll_mat = numpy.asarray([[1, 0, 0],
                               [0, cos(roll), -sin(roll)],
                               [0, sin(roll), cos(roll)]])

    result = yaw_mat @ pitch_mat @ roll_mat
    return result


def transform_item(item):
    image, points, depth = item
    height, width = depth.shape
    label_tensor = torch.zeros((len(common.visible_blocks) + 1, height, width), dtype=torch.uint8)
    for (label, (x, z)) in points:
        label_tensor[label, z, x] = 1
    return image, label_tensor, depth


def add_episode(data_set, ep):
    print('loading ', ep)
    episode_data = load_episode(os.path.join('episodes', ep))
    random.shuffle(episode_data)
    print('len episode', len(episode_data))
    for item in episode_data[:100]:
        data_set.add(item)


if __name__ == '__main__':
    from dataset import MinecraftImageDataset
    from goodpoint import GoodPoint
    import torch.optim as optim
    device = 'cpu'
    train_depth = False
    batch_size = 20

    data_set = MinecraftImageDataset(max_size=700, transform=transform_item)
    from torch.utils.data import DataLoader
    loader = DataLoader(data_set, batch_size=batch_size, shuffle=False)
    # +1 for None
    net = GoodPoint(8, len(common.visible_blocks) + 1, n_channels=3, depth=train_depth).to('cpu')
    model_weights = torch.load('goodpoint.pt')['model']
    net.load_checkpoint(model_weights)
    net.train()
    lr = 0.005
    epochs = 20
    eps = 0.0000001
    optimizer = optim.AdamW(net.parameters(), lr=lr)
    episode_files = os.listdir('episodes')
    leaves = common.visible_block_num['leaves']
    log = common.visible_block_num['log']
    water = common.visible_block_num['water']
    lava = common.visible_block_num['lava']

    show = False

    for ep in episode_files:
        add_episode(data_set, ep)
        if data_set.size() == len(data_set):
            break

    for epoch in range(epochs):
        ep = episode_files[epoch % len(episode_files)]
        print(ep)
        add_episode(data_set, ep)
        for j, batch in enumerate(loader):
            optimizer.zero_grad()
            imgs, (points, depth) = batch
            prediction = net(imgs)
            if train_depth:
                blocks, p_depth = prediction
            else:
                blocks = prediction
            logprob = torch.log(blocks + eps)
            loss_blocks = - logprob[points.nonzero(as_tuple=True)].mean()
            loss = loss_blocks
            if train_depth:
                w = 10
                loss_depth = ((depth.unsqueeze(1) * w - p_depth * w ) ** 2).mean()
                loss += loss_depth
            loss.backward()
            optimizer.step()
            for i in range(1):
                img = imgs[i]
                img0 = (img * 255).numpy().astype(numpy.uint8).transpose(1, 2, 0)
                if show:
                    cv2.imshow('vasya', img0)
                    cv2.imshow('label-log', points[i][log].numpy().astype(numpy.uint8) * 255)
                    cv2.imshow('label-water', points[i][water].numpy().astype(numpy.uint8) * 255)
                    cv2.imshow('label-lava', points[i][lava].numpy().astype(numpy.uint8) * 255)
                    cv2.imshow('depth', (depth[i].numpy() * 255).astype(numpy.uint8))
                    # show grass, leaves, log
                    cv2.imshow('water', (blocks[i][water] * 255).detach().numpy().astype(numpy.uint8))
                    cv2.imshow('depth_p', (p_depth[i][0] * 255).detach().numpy().astype(numpy.uint8))
                    cv2.imshow('lava', (blocks[i][lava] * 255).detach().numpy().astype(numpy.uint8))
                    cv2.imshow('leaves', (blocks[i][leaves] * 255).detach().numpy().astype(numpy.uint8))
                    cv2.imshow('log', (blocks[i][log] * 255).detach().numpy().astype(numpy.uint8))
                    cv2.waitKey(200)
            if j % 10:
                print('loss_blocks', loss)
                if train_depth:
                    print('loss_depth', loss_depth)
        snap = dict()
        snap['model'] = net.state_dict()
        snap['optimizer'] = optimizer.state_dict()
        torch.save(snap, 'goodpoint.pt')
        print(loss)
