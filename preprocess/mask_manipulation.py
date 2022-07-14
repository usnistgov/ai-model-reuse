# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import argparse
import glob
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

"""
This is a helper class for converting .npy files to .png files.
__author__      = "Peter Bajcsy"
__email__ = "peter.bajcsy@nist.gov"
"""
def convert_mask(image_dir, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    os.chdir(image_dir)
    for filename in glob.glob("*.*"):
        if '.npy' in filename:
            print('INFO: loading filename:', filename)
            basename = os.path.basename(filename)
            print('INFO:: basename:', basename)

            img_array = np.load(filename, allow_pickle=True)
            plt.imshow(img_array, cmap="gray")
            out_filename = output_dir + os.path.sep + basename
            # Most backends support png, pdf, ps, eps and svg.
            # according to https://matplotlib.org/2.2.3/api/_as_gen/matplotlib.pyplot.imsave.html
            out_filename = out_filename[:-len(".npy")] + '.png'
            print('INFO: output file:', out_filename)
            matplotlib.image.imsave(out_filename, img_array)
            print(filename)


def main():
    parser = argparse.ArgumentParser(prog='mask_manipulation', description='Script that operates on masks')
    parser.add_argument('--image_dir', type=str, required=True, dest='image_dir', default=None,
                        help='folder path to input images')
    parser.add_argument('--output_dir', type=str, required=True, default=None,
                        help='folder path to saving output converted files')
    args, unknown = parser.parse_known_args()

    if args.image_dir is None:
        print('ERROR: missing input image dir ')
        return

    if args.output_dir is None:
        print('ERROR: missing output image dir ')
        return

    convert_mask(args.image_dir, args.output_dir)


if __name__ == "__main__":
    main()
