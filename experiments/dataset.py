from torch.utils.data import Dataset
import random


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
