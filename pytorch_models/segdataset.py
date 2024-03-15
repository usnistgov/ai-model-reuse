"""
Author: Manpreet Singh Minhas
Contact: msminhas at uwaterloo ca
"""
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset
import skimage
import skimage.io
import torch


class SegmentationDataset(VisionDataset):
    """A PyTorch dataset for image segmentation task.
    The dataset is compatible with torchvision transforms.
    The transforms passed would be applied to both the Images and Masks.
    """

    def __init__(self,
                 root: str,
                 train_image_folder: str,
                 train_mask_folder: str,
                 test_image_folder: str,
                 test_mask_folder: str,
                 transforms: Optional[Callable] = None,
                 seed: int = None,
                 fraction: float = None,
                 subset: str = None,
                 image_color_mode: str = "rgb",
                 mask_color_mode: str = "grayscale",
                 n_classes=4) -> None:

        super().__init__(root, transforms)
        train_image_folder_path = Path(self.root) / train_image_folder
        train_mask_folder_path = Path(self.root) / train_mask_folder
        test_image_folder_path = Path(self.root) / test_image_folder
        test_mask_folder_path = Path(self.root) / test_mask_folder
        self.image_color_mode = image_color_mode
        self.mask_color_mode = mask_color_mode

        self.fraction = fraction
        self.train_image_names = sorted(train_image_folder_path.glob("*"))
        self.train_mask_names = sorted(train_mask_folder_path.glob("*"))
        self.test_image_names = sorted(test_image_folder_path.glob("*"))
        self.test_mask_names = sorted(test_mask_folder_path.glob("*"))
        if subset == "Train":
            weights = [0] * n_classes
            self.image_names = self.train_image_names
            self.mask_names = self.train_mask_names
            for mask_name in self.mask_names:
                image = np.asarray(Image.open(mask_name))
                for i in range(len(weights)):
                    weights[i] += np.count_nonzero(image == i)
            # OLD method:
            # for file in self.mask_names:
            #     with open(file, "rb") as mask_file:
            #         image = np.asarray(Image.open(mask_file))
            #         for h in range(image.shape[0]):
            #             for w in range(image.shape[1]):
            #                 value = image[h][w]
            #                 weights[value] += 1

            self.weights = weights
        #
        else:
            self.image_names = self.test_image_names
            self.mask_names = self.test_mask_names

    def __len__(self) -> int:
        return len(self.image_names)

    # @staticmethod
    # def format_image(x):
    #     # reshape into tensor (CHW)
    #     x = np.transpose(x, [2, 0, 1])
    #     return x

    @staticmethod
    def zscore_normalize(x):
        x = x.astype(np.float32)

        std = np.std(x)
        mv = np.mean(x)
        if std <= 1.0:
            # normalize (but dont divide by zero)
            x = (x - mv)
        else:
            # z-score normalize
            x = (x - mv) / std
        return x

    def __getitem__(self, index: int) -> Any:
        image_path = self.image_names[index]
        mask_path = self.mask_names[index]
        image = skimage.io.imread(image_path)
        image = np.expand_dims(image, 0)
        image = np.concatenate((image, image, image), axis=0)
        mask = skimage.io.imread(mask_path)
        sample = {"image": image, "mask": mask}

        # format the image into a tensor
        # image= self.format_image(image) # no need to re-order channels!!
        image = self.zscore_normalize(image)
        sample['image'] = torch.from_numpy(image)

        # if(image.dtype != "uint8"):
        #     image = image / 65536 #  assumes uint16  --> 256*256
        #     sample['image'] = torch.from_numpy(image)
        # else:
        #     sample['image'] = sample['image'] / 255
        #     sample['image'] = torch.from_numpy(image)

        if (mask.dtype != "uint8" or mask.dtype != "int8"):
            mask = mask.astype('uint8')

        mask_tensor = torch.from_numpy(mask)
        sample['mask'] = mask_tensor
        return sample
