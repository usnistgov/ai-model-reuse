from tifffile import tifffile
import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import transforms

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset
import skimage
import skimage.io
import torch


def _get_image_from_path(image_path, astype=np.float32):
    _img = imread(image_path)
    if astype is not None:
        _img = _img.astype(astype)
    return _img


def return_images_from_paths(image_paths, astype=np.float32):
    """

    :param astype: convert image format. (Not recommended as this can cause many errors down the line)
    :param image_paths:
    :return:
    """

    if isinstance(image_paths, str):
        return _get_image_from_path(image_paths)
    else:
        # print(image_paths)
        assert isinstance(image_paths, (np.ndarray, list))
        image_list = []
        for image_path in image_paths:
            img = _get_image_from_path(image_path, astype)
            image_list.append(img)
        image_list = np.asarray(image_list)
        return image_list


class INFERSegmentationDataset(VisionDataset):
    """A PyTorch dataset for image segmentation task. The dataset is compatible with torchvision transforms.
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
                 n_classes=4) -> None:

        super().__init__(root, transforms)
        train_image_folder_path = Path(self.root) / train_image_folder
        train_mask_folder_path = Path(self.root) / train_mask_folder
        test_image_folder_path = Path(self.root) / test_image_folder
        test_mask_folder_path = Path(self.root) / test_mask_folder

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
                # with open(mask_name, "rb") as mask_file:
                #     image = np.asarray(Image.open(mask_file))
                #
                #     for h in range(image.shape[0]):
                #         for w in range(image.shape[1]):
                #             value = image[h][w]
                #             weights[value] += 1

                image = np.asarray(Image.open(mask_name))
                # print(np.unique(image))
                for i in range(len(weights)):
                    # print(len(weights), i, image.max(), image.min(), flush=True)
                    weights[i] += np.count_nonzero(image == i)

            self.weights = weights
            print("WEIGHTS = ", self.weights)
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
        # image = np.expand_dims(image, 0)
        image = _get_image_from_path(image_path)
        # print("image_shape = ", image.shape)
        # exit()
        # image = np.concatenate((image, image, image), axis=0)
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


class GetDataloader:
    def __init__(self, data_dir, train_image_folder, train_mask_folder, test_image_folder, test_mask_folder, fraction,
                 batch_size, n_classes):
        self.data_dir = data_dir
        self.train_image_folder = train_image_folder
        self.train_mask_folder = train_mask_folder
        self.test_image_folder = test_image_folder
        self.test_mask_folder = test_mask_folder
        self.fraction = fraction
        self.batch_size = batch_size
        self.n_classes = n_classes
        # data_transforms = transforms.Compose([transforms.ToTensor(),
        #                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        data_transforms = transforms.Compose([transforms.ToTensor()])
        image_datasets = {
            x: INFERSegmentationDataset(data_dir,
                                        train_image_folder=train_image_folder,
                                        train_mask_folder=train_mask_folder,
                                        test_image_folder=test_image_folder,
                                        test_mask_folder=test_mask_folder,
                                        fraction=fraction,
                                        subset=x,
                                        transforms=data_transforms,
                                        n_classes=n_classes)
            for x in ['Train', 'Test']
        }
        self.weights = image_datasets['Train'].weights
        dataloaders = {
            x: DataLoader(image_datasets[x],
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=8, drop_last=True)
            for x in ['Train', 'Test']
        }
        self.dataloaders = dataloaders


if __name__ == "__main__":
    test_image = np.zeros((500, 50, 2, 2, 3, 6))
    f_path = "../Data/image_test/test1.tiff"
    test_metadata = {"test_metadata": "this is test metadata",
                     "hello": "hello, hello!"}

    print("hi", test_image.ndim)
