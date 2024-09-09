# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.
# TODO: Remove?
import os
import argparse
import random
from shutil import copyfile
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

"""
This class will split image collections of images and their corresponding masks into
k equal sets. CV stands for cross validation. To allow for repeatable results, seeds can be input to obtain the same 
sets of train and test data. 

    __author__      = "Pushkar Sathe"
    __email__ = "pushkar.sathe@nist.gov"
"""


def split(image_folder, mask_folder, k, shuffle=True, seed=None):
    image_folder = os.path.abspath(image_folder)
    mask_folder = os.path.abspath(mask_folder)
    img_files = [i for i in os.listdir(image_folder)]
    mask_files = [m for m in os.listdir(mask_folder)]
    kf = KFold(n_splits=k, shuffle=shuffle, random_state=seed)
    kf.get_n_splits(X=img_files, y=mask_files)


def segregated_split():
    return


def stratified_split(image_folder, mask_folder, k, shuffle=True, seed=None):
    image_folder = os.path.abspath(image_folder)
    mask_folder = os.path.abspath(mask_folder)
    img_files = [i for i in os.listdir(image_folder)]
    mask_files = [m for m in os.listdir(mask_folder)]
    kf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=seed)
    kf.get_n_splits(X=img_files, y=mask_files)
    return


def main():
    parser = argparse.ArgumentParser(prog='split', description='Script that splits data')
    parser.add_argument('--image_dir', type=str)  # full path of image folder
    parser.add_argument('--mask_dir', type=str)  # full path of mask folder
    # parser.add_argument('--folder_prefix', type=str)  # common prefix for folde
    parser.add_argument('--k', type=float)
    args, unknown = parser.parse_known_args()

    if args.image_dir is None:
        print('ERROR: missing input image folder ')
        return

    split(args.image_dir, args.mask_dir, args.k)



if __name__ == "__main__":
    main()
