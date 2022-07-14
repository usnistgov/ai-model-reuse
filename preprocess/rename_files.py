# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import argparse
import re
from shutil import copyfile

"""
This class will rename file names to make the input image and mask files to be consistent
__author__      = "Peter Bajcsy"
__email__ = "peter.bajcsy@nist.gov"
"""

# this methd was used to fix 2D RPE masks that had values 0 and 255 instead of 0 (boundary) and 1 (inside cell)
def rename_images(image_folder, mask_folder, output_image_folder):
    image_folder = os.path.abspath(image_folder)
    img_files = [f for f in os.listdir(image_folder)]
    mask_folder = os.path.abspath(mask_folder)
    mask_files = [f for f in os.listdir(mask_folder)]

    print('INFO: mask_folder:', mask_files)
    # check that the output folder exists
    if not os.path.exists(output_image_folder):
        os.mkdir(output_image_folder)

    for fmask in mask_files:
        mask_file = os.path.join(mask_folder, fmask)
        out_image_file = os.path.join(output_image_folder, fmask)
        # mask file: P1-W1-TOM_E02_F001_DNA_RPE00<XX>.tif
        matching_segment = 'RPE'
        m = re.search(r'(?<=RPE)\w+', mask_file)
        # print('INFO: mask: m.group(0):', m.group(0))
        # print('INFO: mask: m.group(0):', m)
        mask_str = m.group(0).replace('00', '')
        print('INFO: final mask str:', mask_str)
        if len(mask_str) > 0:
            print('INFO: int(str):', int(mask_str))
            mask_int = int(mask_str)
        else:
            mask_int = 0

        # if os.path.isfile(mask_file) and mask_file.__contains__(matching_segment):

        for fn in img_files:
            image_file = os.path.join(image_folder, fn)
            m = re.search(r'(?<=A01Z)\w+', image_file)
            # print('INFO: image: m.group(0):', m.group(0))
            image_str = m.group(0).replace('C02', '')
            print('INFO: final image str:', image_str)
            if len(image_str) > 0:
                print('INFO: int(str):', int(image_str))
                image_int = int(image_str)
            else:
                image_int = 0

            # match mask file : P1-W1-TOM_E02_F001_DNA_RPE0000.tif
            # with
            # image file : P1-W1-TOM_E02_T0001F001L01A01Z01C02.tif
            if (mask_int + 1) == image_int:
                print('INFO: found a match: image_file:', image_file, ' mask_file:', mask_file)
                copyfile(image_file,out_image_file)
                break



def main():
    parser = argparse.ArgumentParser(prog='split', description='Script that renames image files according to mask files')
    parser.add_argument('--image_folder', type=str, help='full path of image folder')
    parser.add_argument('--mask_folder', type=str, help='full path of mask folder')
    parser.add_argument('--result_image_folder', type=str, help='full path of resulting folder destination')

    args, unknown = parser.parse_known_args()

    if args.image_folder is None:
        print('ERROR: missing input mask folder ')
        return

    rename_images(args.image_folder, args.mask_folder, args.result_image_folder)

if __name__ == "__main__":
    main()

