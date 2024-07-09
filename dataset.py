import os
import random
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import ColorJitter, RandomAffine

import numpy as np
import torch

class ConsistentVideoTransform:
    """Apply the same aggressive transform to each frame in a video."""
    def __init__(self, resize_size, crop_size, transform, validation=False):
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.transform = transform
        self.validation = validation

    def __call__(self, img_sequence):
        if not self.validation:
            # Decide if we should flip or not
            flip = random.random() < 0.5

            # Decide the color jitter parameters for this video
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
            hue = random.uniform(0, 0.1)
            color_jitter = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

            # Decide the affine parameters for this video
            degrees = (-15, 15)
            translate = (random.uniform(0, 0.1), random.uniform(0, 0.1))
            scale_factor = random.uniform(0.9, 1.1)
            scale = (max(0.9, scale_factor - 0.1), min(1.1, scale_factor + 0.1))
            shear = random.uniform(0, 0.2)
            random_affine = RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear)

            # Randomly select a cropping region that will be the same for all frames
            left = random.randint(0, self.resize_size[0] - self.crop_size[0])
            top = random.randint(0, self.resize_size[1] - self.crop_size[1])
            right = left + self.crop_size[0]
            bottom = top + self.crop_size[1]

        transformed_sequence = []
        for img in img_sequence:
            # Resize the image first
            img = img.resize((self.resize_size[0], self.resize_size[1]))

            if not self.validation:
                # Apply Color Jitter
                img = color_jitter(img)

                # Apply Affine Transformation
                img = random_affine(img)

                if flip:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)

                img = img.crop((left, top, right, bottom))
            else:
                # Center crop without random transformations
                img = img.crop((
                    (self.resize_size[0] - self.crop_size[0]) // 2,
                    (self.resize_size[1] - self.crop_size[1]) // 2,
                    (self.resize_size[0] + self.crop_size[0]) // 2,
                    (self.resize_size[1] + self.crop_size[1]) // 2
                ))

            transformed_sequence.append(self.transform(img))
        return transformed_sequence

class RGBFrameDataset(Dataset):
    """Dataset class for 3-channel RGB images."""

    def __init__(self, root, n_frames, transform, is_train):
        super(RGBFrameDataset, self).__init__()
        self.root = root            # where to look for videos
        self.n_frames = n_frames    # number of frames
        self.transform = transform  # transform
        self.is_train = is_train    # train/test splits
        self.labels = ['noaccident', 'accident'] # noaccident (negative) -- accident
        self.folder_paths = [] # paths of folder with the data
        self.label_indices = [] # indexes of labels

        if self.is_train:
            self.root = os.path.join(self.root, 'train') # add train to root
        else:
            self.root = os.path.join(self.root, 'test') # add test to root

        for label_index, label in enumerate(self.labels): # id for a label and then an actual
            label_folder = os.path.join(self.root, label) # folder
            video_folders = [os.path.join(label_folder, name) for name in os.listdir(label_folder)]
            self.folder_paths.extend(video_folders)
            self.label_indices.extend([label_index] * len(video_folders))

        tqdm.write(f'[info] There are {len(self.folder_paths)} videos in the dataset')

    def __len__(self):
        return len(self.folder_paths)

    def __getitem__(self, index):
        """
        :param index: int
        :return result: (n_frames, C=3, H, W), label: 1 for accident, 0 for noaccident
        """
        folder = self.folder_paths[index]
        label = self.label_indices[index]

        jpg_list = os.listdir(folder)
        jpg_list.sort()  # must sort to retain the order

        if len(jpg_list) > self.n_frames:  # there are enough frames
            if self.is_train:
                start = np.random.randint(0, len(jpg_list) - self.n_frames)
            else:
                start = 0
            jpg_list = jpg_list[start:start + self.n_frames]
        elif len(jpg_list) < self.n_frames:  # frames are not enough
            jpg_list += [jpg_list[-1]] * (self.n_frames - len(jpg_list))  # repeat the last frame

        assert len(jpg_list) == self.n_frames
        frames = [Image.open(os.path.join(folder, jpg)) for jpg in jpg_list]
        if isinstance(self.transform, ConsistentVideoTransform):
            frames = self.transform(frames)
        else:
            frames = [self.transform(frame) for frame in frames]
        frames = torch.stack(frames, dim=0)  # (n_frames, C=3, H, W)

        return frames, label

    def collate_fn(self, batch):
        videos = torch.stack([b[0] for b in batch], dim=0)  # (batch_size, n_frames, C=3, H, W)
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)  # (batch_size,)

        return videos, labels