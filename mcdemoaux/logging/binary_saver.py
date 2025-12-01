import os
import cv2
import numpy as np
import h5py
import json


class ImageHDF5Saver:
    def __init__(self, image_folder, image_size = (224, 224), file_types=None):
        self.image_folder = image_folder
        self.image_size = image_size  # (width, height)
        self.file_types = file_types if file_types else ('.jpg', '.jpeg', '.png', '.bmp')

    def _load_images(self):
        image_files = [
            os.path.join(self.image_folder, fname)
            for fname in os.listdir(self.image_folder)
            if fname.lower().endswith(self.file_types)
        ]

        images = []
        for path in image_files:
            try:
                img = cv2.imread(path)
                if img is None:
                    print(f"Warning: Failed to load image: {path}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # img = cv2.resize(img, self.image_size)
                images.append(img.astype(np.uint8))
            except Exception as e:
                print(f"Error loading image {path}: {e}")

        if not images:
            raise ValueError("No images were loaded. Please check the folder and file types.")

        return np.stack(images)

    def save_to_hdf5(self, output_path, dataset_name='images'):
        images_tensor = self._load_images()
        with h5py.File(output_path, 'w') as f:
            f.create_dataset(dataset_name, data=images_tensor, compression="gzip")
        print(f"Saved {len(images_tensor)} images to {output_path} under dataset '{dataset_name}'.")


class ImageWithMetadataHDF5Saver:
    def __init__(self, folder_path, image_size=None, image_exts=None):
        self.folder_path = folder_path
        self.image_size = image_size  # (width, height)
        self.image_exts = image_exts if image_exts else ('.jpg', '.jpeg', '.png')

    def _get_base_filenames(self):
        """Find base names (without extension) that have both image and json."""
        files = os.listdir(self.folder_path)
        images = {os.path.splitext(f)[0] for f in files if f.lower().endswith(self.image_exts)}
        jsons = {os.path.splitext(f)[0] for f in files if f.lower().endswith('.json')}
        return sorted(images & jsons)

    def _get_valid_pairs(self):
        """Find base filenames that have both an image (with extension) and a no-extension metadata file."""
        files = os.listdir(self.folder_path)
        base_names = []

        metadata_files = {f for f in files if '.' not in f}

        # Match images to files with no extension
        for file in files:
            name, ext = os.path.splitext(file)
            if ext.lower() in self.image_exts and name in metadata_files:
                base_names.append(name)

        return sorted(base_names)

    def _load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.image_size is not None:
            img = cv2.resize(img, self.image_size)
        return img.astype(np.uint8)

    def _load_json(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_to_hdf5(self, output_path, image_dataset_name='images', json_dataset_name='metadata'):
        base_names = self._get_base_filenames()
        # base_names = self._get_valid_pairs()
        images = []
        metadata = []

        for name in base_names:
            image_path = os.path.join(self.folder_path, name + '.jpg')
            json_path = os.path.join(self.folder_path, name + '.json')
            # json_path = os.path.join(self.folder_path, name)

            # Try different image extensions if .jpg not found JIC, but we only have .jpg ATM
            if not os.path.exists(image_path):
                for ext in self.image_exts:
                    alt_path = os.path.join(self.folder_path, name + ext)
                    if os.path.exists(alt_path):
                        image_path = alt_path
                        break

            try:
                img = self._load_image(image_path)
                meta = self._load_json(json_path)
                images.append(img)
                metadata.append(meta)
            except Exception as e:
                print(f"Skipping {name}: {e}")

        if not images:
            raise RuntimeError("No valid image+JSON pairs found.")

        images_np = np.stack(images)

        with h5py.File(output_path, 'w') as f:
            f.create_dataset(image_dataset_name, data=images_np, compression='gzip')

            json_strings = [json.dumps(m, ensure_ascii=False) for m in metadata]
            string_dtype = h5py.string_dtype(encoding='utf-8')
            f.create_dataset(json_dataset_name, data=json_strings, dtype=string_dtype)

        print(f"Saved {len(images_np)} items to {output_path}")
