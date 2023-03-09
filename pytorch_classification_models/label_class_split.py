# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import argparse
from shutil import copyfile

"""
This class will gather images for the same class labels from subfolders
and split those image collections into two folders:
train
val
according to the split fraction

__author__      = "Peter Bajcsy"
__email__ = "peter.bajcsy@nist.gov"
"""
def sample_rename_split(input_dir, number_samples_per_class, fraction, out_dir):
    # prepare output folders
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    train_dir = os.path.join(out_dir, 'train')
    val_dir = os.path.join(out_dir, 'val')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)

    # # counter
    # counter = 0
    for parent_dirname in os.walk(input_dir):
        # ignore the mask folders
        if os.path.isdir(parent_dirname[0]) and not parent_dirname[0].endswith('_mask'):
            print('DEBUG: parent_dirname = ', parent_dirname[0])
            # restrict the number of folders to 5
            # if counter > 5:
            #     return
            # counter = counter + 1

            for k in range(0, len(parent_dirname[1])):
                if parent_dirname[1][k].endswith('_mask'):
                    continue
                child_subdir = os.path.join(parent_dirname[0],parent_dirname[1][k])
                print('DEBUG: child_subdir = ', child_subdir)
                for child_dirname in os.walk(child_subdir):
                    for j in range(0, len(child_dirname[1])):
                        class_subdir = os.path.join(child_dirname[0], child_dirname[1][j])
                        # print('DEBUG: class_subdir=', class_subdir)
                        class_sample_count = 0
                        # decide how many image samples go to train vs validation
                        available_samples = len(os.listdir(class_subdir))
                        if available_samples < number_samples_per_class:
                            train_class_sample_count = int(fraction * available_samples)
                        else:
                            train_class_sample_count = int(fraction * number_samples_per_class)

                        for fn in os.listdir(class_subdir):
                            # take only number_samples_per_class from each class subdirectory
                            if class_sample_count >= number_samples_per_class:
                                break

                            file = os.path.join(class_subdir, fn)
                            if os.path.isfile(file):
                                # split the images to be copied to train or validation directories
                                if class_sample_count < train_class_sample_count:
                                    dest_class_subdir = os.path.join(train_dir, child_dirname[1][j])
                                else:
                                    dest_class_subdir = os.path.join(val_dir, child_dirname[1][j])

                                # print('DEBUG: dest_class_subdir=', dest_class_subdir)
                                if not os.path.exists(dest_class_subdir):
                                    os.mkdir(dest_class_subdir)
                                out_filename = parent_dirname[1][k] + "_" + fn
                                # print('DEBUG: out_filename=', out_filename)
                                copyfile(file, "{}/{}".format(dest_class_subdir, out_filename))
                                class_sample_count = class_sample_count + 1


def main():
    parser = argparse.ArgumentParser(prog='split', description='Script that samples images for the same class label across folders, renames then, and splits data')
    parser.add_argument('--input_dir', type=str, help='input directory', required=True)
    parser.add_argument('--output_dir', type=str, help='output directory', required=True)
    parser.add_argument('--fraction', type=float, help='float number in [0,1] defininf train:val ratio', required=True)
    parser.add_argument('--num_samples_class', type=int, help='number of samples selected from each class folder', required=True)
    args, unknown = parser.parse_known_args()

    if args.input_dir is None:
        print('ERROR: missing input folder ')
        return
    if args.output_dir is None:
        print('ERROR: missing output folder ')
        return

    number_samples_per_class = 3
    sample_rename_split(args.input_dir,args.num_samples_class, args.fraction, args.output_dir)


if __name__ == "__main__":
    main()

