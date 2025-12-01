import h5py
import numpy as np
import json


class ImageHDF5Loader:
    def __init__(self, hdf5_path, dataset_name='images'):
        self.hdf5_path = hdf5_path
        self.dataset_name = dataset_name
        self._file = None
        self._dataset = None
        self._open()

    def _open(self):
        self._file = h5py.File(self.hdf5_path, 'r')
        if self.dataset_name not in self._file:
            raise KeyError(f"Dataset '{self.dataset_name}' not found in file '{self.hdf5_path}'.")
        self._dataset = self._file[self.dataset_name]

    def get_shape(self):
        """Return shape: (num_images, height, width, channels)"""
        return self._dataset.shape

    def get_batch(self, start, end):
        if not (0 <= start < end <= len(self._dataset)):
            raise IndexError("Invalid slice range.")
        return self._dataset[start:end]

    def get_all(self):
        return self._dataset[:]

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
            self._dataset = None

    def __del__(self):
        self.close()


class ImageWithMetadataHDF5Loader:
    def __init__(self, hdf5_path, image_dataset_name='images', json_dataset_name='metadata'):
        self.hdf5_path = hdf5_path
        self.image_dataset_name = image_dataset_name
        self.json_dataset_name = json_dataset_name
        self._file = None
        self._images = None
        self._metadata = None
        self._open()

    def _open(self):
        self._file = h5py.File(self.hdf5_path, 'r')

        if self.image_dataset_name not in self._file:
            raise KeyError(f"Dataset '{self.image_dataset_name}' not found in file.")
        if self.json_dataset_name not in self._file:
            raise KeyError(f"Dataset '{self.json_dataset_name}' not found in file.")

        self._images = self._file[self.image_dataset_name]
        self._metadata = self._file[self.json_dataset_name]

    def get_count(self):
        return len(self._images)

    def get_all(self):
        images = self._images[:]
        metadata = [json.loads(s) for s in self._metadata[:]]
        return images, metadata

    def get_batch(self, start, end):
        if not (0 <= start < end <= len(self._images)):
            raise IndexError("Invalid range.")
        images = self._images[start:end]
        metadata = [json.loads(s) for s in self._metadata[start:end]]
        return images, metadata

    def get_item(self, index):
        if not (0 <= index < len(self._images)):
            raise IndexError("Index out of bounds.")
        image = self._images[index]
        metadata = json.loads(self._metadata[index])
        return image, metadata

    def close(self):
        if self._file:
            self._file.close()
            self._file = None
            self._images = None
            self._metadata = None

    def __del__(self):
        self.close()
