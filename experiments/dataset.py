import cv2
import os
import os.path
import random
from torch.utils.data import Dataset


class MinecraftImageDataset(Dataset):
    def __init__(self, max_size, transform=None):
        self.items = []
        self.max_size = max_size
        self.transform = transform
        self.position = 0

    def add(self, element):
        if len(self.items) < self.max_size:
            self.items.append(element)
        else:
            self.position = random.randint(0, self.max_size - 1)
            self.items[self.position] = element

    def __len__(self):
        return self.max_size

    def __getitem__(self, idx):
        idx = idx % len(self.items)
        item = self.items[idx]
        if self.transform:
            item = self.transform(item)
        image = item[0]
        label = item[1:]
        return image, label

    def size(self):
        return len(self.items)

IMG = 0
SEGM = 1

class MinecraftSegmentation(Dataset):
    def __init__(self, imagedir, transform=None):
        self.imagedir = imagedir
        self.pairs = self._load()
        self.transform = transform

    def _load(self):
        pairs = []
        for f in os.listdir(self.imagedir):
            if f.startswith('img') and f.endswith('.png'):
                segm_f = f.replace('img', 'seg')
                pairs.append((f, segm_f))
        return pairs

    def __getitem__(self, idx):
        item0 = cv2.imread(os.path.join(self.imagedir, self.pairs[idx][IMG]))
        item1 = cv2.imread(os.path.join(self.imagedir, self.pairs[idx][SEGM]))
        assert item0 is not None
        assert item1 is not None
        if self.transform:
            item0, item1 = self.transform((item0, item1))
        return item0, item1

    def __len__(self):
        return len(self.pairs)

