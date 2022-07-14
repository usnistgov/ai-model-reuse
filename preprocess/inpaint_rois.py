# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import argparse
import time
from copy import deepcopy

import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

"""
This class will use inpainting to fuse all regions of interest (ROIs) with the background
The inpainting method is using opencv methods 
 
https://docs.opencv.org/4.x/df/d3d/tutorial_py_inpainting.html
https://www.pyimagesearch.com/2020/05/18/image-inpainting-with-opencv-and-python/

    Bertalmio, Marcelo, Andrea L. Bertozzi, and Guillermo Sapiro. "Navier-stokes, fluid dynamics, and image and video inpainting." In Computer Vision and Pattern Recognition, 2001. CVPR 2001. Proceedings of the 2001 IEEE Computer Society Conference on, vol. 1, pp. I-355. IEEE, 2001.
    Telea, Alexandru. "An image inpainting technique based on the fast marching method." Journal of graphics tools 9.1 (2004): 23-34.


    cv2.INPAINT_TELEA
    : An image inpainting technique based on the fast marching method (Telea, 2004)
    cv2.INPAINT_NS
    : Navier-stokes, Fluid dynamics, and image and video inpainting (Bertalmío et al., 2001)



"""


def inpaint_rois(image_folder, mask_folder, output_image_folder):

    image_folder = os.path.abspath(image_folder)
    image_files = [f for f in os.listdir(image_folder)]

    mask_folder = os.path.abspath(mask_folder)
    mask_files = [f for f in os.listdir(mask_folder)]
    print('INFO: mask_folder:', mask_files)
    # threshold for binarizing mask labels
    t = 0
    # check that the output folder exists
    if not os.path.exists(output_image_folder):
        os.mkdir(output_image_folder)

    for fn in mask_files:
        image_file = os.path.join(image_folder, fn)
        mask_file = os.path.join(mask_folder, fn)
        out_image_file = os.path.join(output_image_folder, fn)
        basename = fn.rsplit('.', 1) # split on the last occurrence of the delimiter

        if os.path.isfile(image_file) and os.path.isfile(mask_file):
            start = time.time()
            # image = skimage.io.imread(fname=image_file)
            # mask = skimage.io.imread(fname=mask_file)

            # see the flags at https://www.geeksforgeeks.org/python-opencv-cv2-imread-method/
            image = cv.imread(image_file, -1) # 0 or cv.IMREAD_GRAYSCALE, -1 is unchanged
            mask = cv.imread(mask_file, 0)
            print('image.shape:', image.shape)
            print('mask.shape:', mask.shape)

            dst = cv.inpaint(image, mask, 7, cv.INPAINT_TELEA)
            #dst = cv.inpaint(image, mask, 7, cv.INPAINT_NS)

            # TODO if input is RGB then   convert amd save gray !!!
            #gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)

            #skimage.io.imsave(fname=out_image_file, arr=dst)
            cv.imwrite(out_image_file, dst)



def main():
    parser = argparse.ArgumentParser(prog='inpaint_rois', description='Script that inpaints the ROIs based on nbh pixels')
    parser.add_argument('--image_dir', type=str, help='full path of image folder')
    parser.add_argument('--mask_dir', type=str, help='full path of mask folder')
    parser.add_argument('--output_dir', type=str, help='full path to output folder destination')

    args, unknown = parser.parse_known_args()

    if args.image_dir is None:
        print('ERROR: missing input image folder ')
        return
    if args.mask_dir is None:
        print('ERROR: missing input mask folder ')
        return
    if args.output_dir is None:
        print('ERROR: missing outputimage folder ')
        return

    inpaint_rois(args.image_dir, args.mask_dir, args.output_dir)

if __name__ == "__main__":
    main()

