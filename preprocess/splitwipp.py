# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import argparse
import random
from shutil import copyfile

"""
This class will split image collections of images and their corresponding masks into
four folders:
train_images
train_masks
test_images
test_masks
according to the split fraction

"""


def split(image_folder, mask_folder, train_image_folder, train_mask_folder, test_image_folder, test_mask_folder,
          fraction):
    # TODO: Currently naming files as ".tif" and ".tiff" doesnt match. Add handling for this and other extension types
    image_folder = os.path.abspath(image_folder)
    mask_folder = os.path.abspath(mask_folder)
    img_files = [f for f in os.listdir(mask_folder)]
    print(len(img_files))
    random.shuffle(img_files)
    idx = int(fraction * len(img_files))
    train_img_files = img_files[0:idx]
    test_img_files = img_files[idx:]
    print(train_img_files, test_img_files)
    for fn in train_img_files:
        file = os.path.join(image_folder, fn)
        mask_file = os.path.join(mask_folder, fn)
        print(image_folder, mask_folder, fn)
        if os.path.isfile(file) and os.path.isfile(mask_file):
            if not os.path.exists(train_image_folder):
                os.mkdir(train_image_folder)
            if not os.path.exists(train_mask_folder):
                os.mkdir(train_mask_folder)
            copyfile(file, "{}/{}".format(train_image_folder, fn))
            copyfile(mask_file, "{}/{}".format(train_mask_folder, fn))

    for fn in test_img_files:
        file = os.path.join(image_folder, fn)
        mask_file = os.path.join(mask_folder, fn)
        if os.path.isfile(file) and os.path.isfile(mask_file):
            if not os.path.exists(test_image_folder):
                os.mkdir(test_image_folder)
            if not os.path.exists(test_mask_folder):
                os.mkdir(test_mask_folder)
            copyfile(file, "{}/{}".format(test_image_folder, fn))
            copyfile(mask_file, "{}/{}".format(test_mask_folder, fn))


def main():
    parser = argparse.ArgumentParser(prog='split', description='Script that splits data')
    parser.add_argument('-i', '--image_dir', type=str, help="full path of image folder")  #
    parser.add_argument('-m', '--mask_dir', type=str, help="full path of mask folder")
    parser.add_argument('-f', '--fraction', type=float, help="fraction in train")
    parser.add_argument('-o', '--outputroot', type=str, help="root path for output")
    parser.add_argument('-tri', '--trainImageDir', type=str,
                        help="relative path to train image folder destination from outputroot")
    parser.add_argument('-tei', '--testImageDir', type=str,
                        help="relative path to image folder destination from outputroot")
    parser.add_argument('-trm', '--trainMaskDir', type=str,
                        help="relative path to train mask folder destination from outputroot")
    parser.add_argument('-tem', '--testMaskDir', type=str,
                        help="relative path to test mask folder destination from outputroot")
    args, unknown = parser.parse_known_args()

    if args.image_dir is None:
        print('ERROR: missing input image folder ')
        return
    print(args)
    # print(unknown)
    trainImageDir = os.path.join(args.outputroot, args.trainImageDir)
    trainMaskDir = os.path.join(args.outputroot, args.trainMaskDir)
    testImageDir = os.path.join(args.outputroot, args.testImageDir)
    testMaskDir = os.path.join(args.outputroot, args.testMaskDir)

    print(args.image_dir, args.mask_dir, trainImageDir, trainMaskDir, testImageDir, testMaskDir)
    split(args.image_dir, args.mask_dir, trainImageDir, trainMaskDir, testImageDir, testMaskDir, args.fraction)


if __name__ == "__main__":
    main()
