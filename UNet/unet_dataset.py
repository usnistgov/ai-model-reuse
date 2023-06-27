# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import numpy as np

import torch
import torch.utils.data

import random
import lmdb
from torch.utils.data import WeightedRandomSampler

from isg_ai_pb2 import ImageMaskPair

from PIL import Image
import torchvision
from torchvision import transforms
class UnetDataset(torch.utils.data.Dataset):
    """
    data set for UNet, image-mask pair_param
    """

    def __init__(self, lmdb_filepath, nb_classes, augment=False):
        self.lmdb_filepath = lmdb_filepath

        self.augment = augment
        self.nb_classes = nb_classes
        self.__init_database()
        self.lmdb_txn = self.lmdb_env.begin(write=False)  # hopefully a shared instance across all threads

    def __init_database(self):
        random.seed()

        # get a list of keys from the lmdb
        self.keys_flat = list()
        self.keys = list()
        self.keys.append(list())  # there will always be at least one class

        self.lmdb_env = lmdb.open(self.lmdb_filepath, map_size=int(2e10), readonly=True)  # 1e10 is 10 GB

        #present_classes_str_flat = list()

        datum = ImageMaskPair()
        print('Initializing image database')

        with self.lmdb_env.begin(write=False) as lmdb_txn:
            cursor = lmdb_txn.cursor()

            # move cursor to the first element
            cursor.first()
            # get the first serialized value from the database and convert from serialized representation
            datum.ParseFromString(cursor.value())
            # record the image size
            self.image_size = [datum.img_height, datum.img_width, datum.channels]

            # iterate over the database getting the keys
            for key, val in cursor:
                self.keys_flat.append(key)

        print('Dataset has {} examples'.format(len(self.keys_flat)))

    def get_image_count(self):
        # tie epoch size to the number of images
        return int(len(self.keys_flat))

    def get_image_size(self):
        #return self.image_size #need to change
        return [self.image_size[0], self.image_size[1], 3]
    def get_image_tensor_shape(self):
        # HWC to CHW
        #return [self.image_size[2], self.image_size[0], self.image_size[1]]
        #return [self.image_size[0], self.image_size[1], self.image_size[2]] #change
        return [self.image_size[0], self.image_size[1], 3]
    def get_label_tensor_shape(self):
        return [self.image_size[0], self.image_size[1]]

    def get_number_classes(self):
        return self.number_classes

    def get_image_shape(self):
        return self.get_image_size()

    def __len__(self):
        return len(self.keys_flat)

    @staticmethod
    def format_image(x):
        # reshape into tensor (CHW)
        x = np.transpose(x, [2, 0, 1])
        return x

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

    def __getitem__(self, index):
        datum = ImageMaskPair()  # create a datum for decoding serialized caffe_pb2 objects
        fn = self.keys_flat[index]

        # extract the serialized image from the database
        value = self.lmdb_txn.get(fn)
        # convert from serialized representation
        datum.ParseFromString(value)

        # convert from string to numpy array
        img = np.fromstring(datum.image, dtype=datum.img_type)
        # reshape the numpy array using the dimensions recorded in the datum
        img = img.reshape((datum.img_height, datum.img_width, datum.channels))
        if np.any(img.shape != np.asarray(self.image_size)):
            raise RuntimeError("Encountered unexpected image shape from database. Expected {}. Found {}.".format(self.image_size, img.shape))
        # convert from string to numpy array
        M = np.fromstring(datum.mask, dtype=datum.mask_type)
        # reshape the numpy array using the dimensions recorded in the datum
        M = M.reshape(datum.img_height, datum.img_width, datum.channels)
        # format the image into a tensor
        img = self.format_image(img) # must re-order channels to match pytorch convention
        img = self.zscore_normalize(img)

        #M = M.astype(np.int32)
        # convert to a one-hot (HWC) representation
        # h, w, d = M.shape
        # M = M.reshape(-1)
        # fM = np.zeros((len(M), self.nb_classes), dtype=np.int32)
        # fM[np.arange(len(M)), M] = 1
        # fM = fM.reshape((h, w, d, self.nb_classes))
        ###### numpy concatenate image,image,image
        img = np.concatenate((img, img, img), axis=0)
        # print(img.shape)
        # print(" ")
        img = torch.from_numpy(img)

        # TODO to support 16 BPP masks instead of dividing by 255 !!!
        #M = M / 255
        if(M.dtype != "uint8" or M.dtype != "int8"):
            M = M.astype('uint8')

        M = torch.from_numpy(M)

        return img, M