import os
import glob
from skimage.color import rgb2lab
from config import Config

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

CONFIG = Config()
class CustomDataset(Dataset):
    def __init__(self, root_folder: str, split: str='train', transform=None):
        all_gray_images = list(glob.glob(os.path.join(root_folder, 'gray', '*.jpg')))
        all_rgb_images = list(glob.glob(os.path.join(root_folder, 'color', '*.jpg')))
        n_train_images = int(0.8 * len(all_gray_images))
        self.transform = transform
        
        if split == 'train':
            self.gray_images = all_gray_images[:n_train_images]
            self.rgb_images = all_rgb_images[:n_train_images]
        elif split == 'val':
            self.gray_images = all_gray_images[n_train_images:]
            self.rgb_images = all_rgb_images[n_train_images:]
        else:
            raise ValueError(f'Split must be either "train" or "val", got {split}')
    
    def __len__(self):
        return len(self.gray_images)
    
    def __getitem__(self, idx):
        grayscale_img = np.array(Image.open(self.gray_images[idx]).convert('L'))
        rgb_img = np.array(Image.open(self.rgb_images[idx]).convert('RGB'))
        
        if self.transform:
            augmented = self.transform(image=grayscale_img, colored=rgb_img)
            grayscale_img = augmented['image']
            rgb_img = augmented['colored']

        if CONFIG.output_format == 'ab':
            # Convert the RGB image to LAB
            lab_img = rgb2lab(rgb_img.numpy(), channel_axis=0).astype('float32')
            lab_img = torch.from_numpy(lab_img)
        
            # Preprocessing step
            gray = lab_img[0, :, :].unsqueeze(0) / 50. - 1.  # Normalize L channel to [-1, 1]
            colored = lab_img[1:, :, :] / 128.  # Normalize AB channels to [-1, 1]
        else:
            # Preprocessing step
            gray = grayscale_img / 127.5 - 1.0  # Normalize grayscale image to [-1, 1]
            colored = rgb_img / 127.5 - 1.0  # Normalize rgb image to [-1, 1]
        return {'gray': gray, 'colored': colored}
