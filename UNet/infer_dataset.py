# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import sys

if sys.version_info[0] < 3:
    raise Exception('Python3 required')

import skimage.io
import numpy as np
import os
import skimage
import skimage.transform
import argparse
import unet_dataset
import build_lmdb
import pandas as pd
import torch

from pathlib import Path


# import re
# import shutil
# import lmdb
# import random

# Load images, then loop over loaded images and normalize them using z score (unet_dataset.py normalize_zcore)
# Pass image to the model
# save result as a segmented image

# same preprocessing for training (3 channels)


#    img = skimage.io.imread(test_loader.dataset.list_IDs[i])
#     img = preprocess_round2(img, model_name)
def confusion_matrix(pred, mask):
    matrix = np.zeros((4, 4))
    for h in range(pred.shape[0]):
        for w in range(pred.shape[1]):
            matrix[pred[h][w]][mask[h][w]] += 1

    return matrix


'''
this method runs inference and reports confusion matrix base don the provided mask

'''

''' 
run inference with model_filemath on images in image_filepath and 
save results in output_dir
'''


def inference(model_filepath, image_filepath, output_dir):
    model = torch.load(model_filepath)
    model.eval()

    file_array = []
    for filename in os.listdir(image_filepath):
        filepath = os.path.join(image_filepath, filename)
        file_array.append(filepath)

    for i in range(len(file_array)):
        img = skimage.io.imread(file_array[i])
        img = build_lmdb.enforce_size_multiple(img)
        if type(img) is not np.ndarray:
            raise Exception("Img must be numpy array to store into db")
        if len(img.shape) > 3:
            raise Exception("Img must be 2D or 3D [HW, or HWC] format")
        if len(img.shape) < 2:
            raise Exception("Img must be 2D or 3D [HW, or HWC] format")
        if len(img.shape) == 2:
            # make a 3D array
            img = img.reshape((img.shape[0], img.shape[1], 1))

        # img = unet_dataset.UnetDataset.zscore_normalize(img)
        # img = img.reshape(img.shape[0], img.shape[1], 1)
        img = unet_dataset.UnetDataset.format_image(img)  # re-order channels to meet the pytorch CHW order
        img = np.concatenate((img, img, img), axis=0)

        # TypeError: can't convert np.ndarray of type numpy.uint16. The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint8, and bool.
        img = img.astype(np.float32)

        img = torch.from_numpy(img)
        img = torch.unsqueeze(img, 0)
        # print(img.shape)
        pred = model.forward(img)
        # print(pred.shape)
        pred = torch.squeeze(pred, 0)
        pred = torch.argmax(pred, 0)
        # print(pred.shape)
        print('-----------------------------------')
        # name_start = file_array[i].find('S')
        # name = file_array[i][name_start:100]
        pred = pred.cpu().detach().numpy().astype(np.uint8)
        basename = os.path.basename(file_array[i])
        # i += 1
        output_fullpath = str(output_dir) + "/pred_{}".format(basename)
        print('output_fullpath:', output_fullpath)
        skimage.io.imsave(output_fullpath, pred)


def main():
    # # Setup the Argument parsing
    # parser = argparse.ArgumentParser(prog='inference', description='Script which performs inference using a unet model')
    # parser.add_argument('--model_filepath', type=str)
    # parser.add_argument('--image_filepath', type=str)
    # #parser.add_argument('--mask_filepath', type=str)
    # #parser.add_argument('--epoch', type=str)
    # args = parser.parse_args()
    # print('hi')
    # print('args %s \n %s \n' % (args.model_filepath, args.image_filepath))
    # inference(args.model_filepath, args.image_filepath, args.mask_filepath, args.epoch)

    parser = argparse.ArgumentParser(prog='inference', description='Script which performs inference using a unet model')
    parser.add_argument('--model_filepath',
                        type=str)  # this should be FULL PATH of the weights.pt file you generated from unet train.py
    parser.add_argument('--image_dirpath', type=str)  # this should be FULL PATH of your images to perform inference on
    parser.add_argument('--output_dirpath',
                        type=str)  # this should be FULL PATH of where you want to store your predictions

    args, unknown = parser.parse_known_args()

    if args.model_filepath is None:
        print('ERROR: missing input model_filepath')
        return

    print('model_filepath:', args.model_filepath)
    print('image_dirpath:', args.image_dirpath)
    print('output_dirpath:', args.output_dirpath)

    if not Path(args.output_dirpath).exists():
        Path(args.output_dirpath).mkdir()

    inference(args.model_filepath, args.image_dirpath, args.output_dirpath)


if __name__ == "__main__":
    main()
