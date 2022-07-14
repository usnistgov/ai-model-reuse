# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import argparse
import numpy as np
import skimage.io

"""
This class will remap mask labels to 0 background and 1 for foreground
__author__      = "Peter Bajcsy"
__email__ = "peter.bajcsy@nist.gov"
"""

# this methd was used to fix 2D RPE masks that had values 0 and 255 instead of 0 (boundary) and 1 (inside cell)
def binarize(mask_folder, output_mask_folder):
    mask_folder = os.path.abspath(mask_folder)
    mask_files = [f for f in os.listdir(mask_folder)]
    print('INFO: mask_folder:', mask_files)
    # threshold for binarizing mask labels
    t = 0
    # check that the output folder exists
    if not os.path.exists(output_mask_folder):
        os.mkdir(output_mask_folder)

    for fn in mask_files:
        mask_file = os.path.join(mask_folder, fn)
        out_mask_file = os.path.join(output_mask_folder, fn)
        if os.path.isfile(mask_file):

            mask = skimage.io.imread(fname=mask_file)
            print('image.shape:', mask.shape)
            # create 2D array and set it to 0
            rows, cols = (mask.shape[0], mask.shape[1])
            result_mask = [[0 for i in range(cols)] for j in range(rows)]

            for i in range(0, cols):
                for j in range(0, rows):
                    if mask[j][i] > 0:
                        result_mask[j][i] = 1

            result_mask = np.array(result_mask, dtype=np.uint8)
            skimage.io.imsave(fname=out_mask_file, arr=result_mask)

# this class re-maps the labels from 3D RPE project with
# input labels: 128 - BKG, 0 - interior of nuclei and 255 boundary of nuclei
def map_labels(mask_folder, output_mask_folder):
    mask_folder = os.path.abspath(mask_folder)
    mask_files = [f for f in os.listdir(mask_folder)]
    print('INFO: mask_folder:', mask_files)

    # check that the output folder exists
    if not os.path.exists(output_mask_folder):
        os.mkdir(output_mask_folder)

    for fn in mask_files:
        mask_file = os.path.join(mask_folder, fn)
        out_mask_file = os.path.join(output_mask_folder, fn)
        if os.path.isfile(mask_file):

            mask = skimage.io.imread(fname=mask_file)
            print('image.shape:', mask.shape)
            # create 2D array and set it to 0
            rows, cols = (mask.shape[0], mask.shape[1])
            result_mask = [[0 for i in range(cols)] for j in range(rows)]

            for i in range(0, cols):
                for j in range(0, rows):
                    if mask[j][i] != 128:
                        result_mask[j][i] = 1

            result_mask = np.array(result_mask, dtype=np.uint8)
            skimage.io.imsave(fname=out_mask_file, arr=result_mask)



def main():
    parser = argparse.ArgumentParser(prog='split', description='Script that splits data')
    parser.add_argument('--mask_folder', type=str, help='full path of mask folder')
    parser.add_argument('--result_folder', type=str, help='full path of resulting folder destination')

    args, unknown = parser.parse_known_args()

    if args.mask_folder is None:
        print('ERROR: missing input mask folder ')
        return

    binarize(args.mask_folder, args.result_folder)
    #map_labels(args.mask_folder, args.result_folder)

if __name__ == "__main__":
    main()

