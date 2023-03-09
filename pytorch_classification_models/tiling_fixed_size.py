# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.
import numpy
from PIL import Image
import os
import argparse

"""
This class will tile each image in the input folder into tiles of fixed dimensions and save then in the output folder.

This was developed for the Kaggle PANDA biopsy images - it automatically analyzes mask tiles and assign Gleson score
based on the majority label in a tile
The output tile names follow the naming convention class_GleasonScore_rowID_colID.png and the tiles are placed in 
subdirectories denoting the Gleason score  

The arguments refer to the dimensions along x- and y-axes (xDim and yDim). 
Thus, an input image of size 1024 x 1024 will be cut into 16 tiles of size 256 x 256 with arguments
--xDim 256 --yDim 225

__author__      = "Peter Bajcsy"
__email__ = "peter.bajcsy@nist.gov"
"""

def count_digits(n):
    count = 0
    while (n > 0):
        count = count + 1
        n = n // 10
    print('The number of digits in the number:', n, ' are:', count)
    return count

def tile(mask_file, width, height, mask_name, image_dir, output_dir):
    # in order to open images larger than limit of 178956970 pixels
    # based on https://stackoverflow.com/questions/56174099/how-to-load-images-larger-than-max-image-pixels-with-pil
    Image.MAX_IMAGE_PIXELS = None

    folder_name, file_extension = os.path.splitext(mask_name)
    target_file_extension = '.png'
    mask_output_tiledir = os.path.join(output_dir,folder_name)
    print('INFO: mask output folder with tiles:', mask_output_tiledir, ' for mask image:', mask_name)
    if not os.path.exists(mask_output_tiledir):
        os.mkdir(mask_output_tiledir)

    # remove _mask from the imag_name
    intensity_name = str(folder_name).split('_')[0]
    if intensity_name is None:
        print('ERROR: could not derive the name from the mask_name:', mask_name )
        return

    intensity_output_tiledir = os.path.join(output_dir,intensity_name)
    if len(str(folder_name).split('_')) < 2:
        print('INFO: mask and intensity must have unique tile directories')
        intensity_output_tiledir = intensity_output_tiledir + '_img'

    if not os.path.exists(intensity_output_tiledir):
        os.mkdir(intensity_output_tiledir)

    intensity_file = os.path.join(image_dir,(intensity_name + file_extension))
    print('INFO: intensity_file:', intensity_file)
    if not os.path.isfile(intensity_file):
        print('ERROR: there is no intensity image that matches the file name of the mask image:', mask_name )
        return

    print('INFO: opening mask image:', mask_file)
    mask_im = Image.open(mask_file)

    imgwidth, imgheight = mask_im.size
    yPieces = imgheight // height
    xPieces = imgwidth // width
    xnum_digits = count_digits(xPieces)
    ynum_digits = count_digits(yPieces)

    # open the intensity image
    intensity_im = Image.open(intensity_file)
    # check that the size is the same
    temp_width, temp_height = intensity_im.size
    if temp_width != imgwidth or temp_height != imgheight:
        print('ERROR: mask and intensity images do not have a matching size: intensity =', intensity_im.size, ' mask:', mask_im.size )
        return

    current_height = 0
    for i in range(0, yPieces):
        row_id = str(i)
        row_id = row_id.rjust(ynum_digits, '0')
        current_width = 0
        for j in range(0, xPieces):
            box = (current_width, current_height, (current_width + width), (current_height +  height) )
            # include all pixels in the image by modifying the size of the last tile
            if i == yPieces-1:
                box = (current_width, current_height, (current_width + width), imgheight)
            if j == xPieces-1:
                box = (current_width, current_height, imgwidth, (current_height +  height) )
            if i == yPieces-1 and j == xPieces-1:
                box = (current_width, current_height, imgwidth, imgheight)


            a = mask_im.crop(box)
            # analyze the labels to determine the Gleason score
            r, g, b = a.split()
            #print('DEBUG: len(histogram):', len(r.histogram()))
            ### 256 ###
            # print('DEBUG: histogram:', r.histogram())
            max_gleason = numpy.argmax(r.histogram())
            #print('DEBUG: histogram:', r.histogram(),'\n max_gleason:', max_gleason)

            col_id = str(j)
            col_id = col_id.rjust(xnum_digits, '0')
            print('DEBUG: row_id:', row_id, ' col_id:', col_id, ' max_gleason:', max_gleason)

            # place the files into sub-folders depending on the gleason score
            mask_output_tile_gleasondir = os.path.join(mask_output_tiledir, str(max_gleason))
            if not os.path.exists(mask_output_tile_gleasondir):
                os.mkdir(mask_output_tile_gleasondir)

            output_file = mask_output_tile_gleasondir + "/class_" + str(max_gleason) + "_example_r" + row_id + "_c" + col_id + target_file_extension
            r.save(output_file, 'png', icc_profile=a.info.get('icc_profile'))
            # save the intensity image
            a1 = intensity_im.crop(box)
            # place the files into sub-folders depending on the gleason score
            intensity_output_tile_gleasondir = os.path.join(intensity_output_tiledir, str(max_gleason))
            if not os.path.exists(intensity_output_tile_gleasondir):
                os.mkdir(intensity_output_tile_gleasondir)
            output_file = intensity_output_tile_gleasondir + "/class_" + str(max_gleason) + "_example_r" + row_id + "_c" + col_id + target_file_extension
            a1.save(output_file, 'png', icc_profile=a1.info.get('icc_profile'))

            current_width = current_width + width

        # increment height
        current_height = current_height + height

def tile_batch(image_dir, mask_dir, output_dir, xDim, yDim):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    mask_file_array = []
    filename_array = []
    for filename in os.listdir(mask_dir):
        filepath = os.path.join(mask_dir, filename)
        mask_file_array.append(filepath)
        filename_array.append(filename)

    i = 0
    for mask_file in mask_file_array:
        tile(mask_file,xDim,yDim,filename_array[i], image_dir, output_dir)
        i += 1

def main():
    parser = argparse.ArgumentParser(prog='split', description='Script that tiles data')
    parser.add_argument('--image_dir', type=str, required=True, help='folder path to input images')
    parser.add_argument('--mask_dir', type=str, required=True, help='folder path to mask images')
    parser.add_argument('--output_dir', type=str, required=True, help='folder path to saving output tiles')
    parser.add_argument('--xDim', type=int, help='tile dimension in pixels along x-axis')
    parser.add_argument('--yDim', type=int, help='tile dimension in pixels along y-axis')
    args, unknown = parser.parse_known_args()

    if args.image_dir is None:
        print('ERROR: missing input image dir ')
        return

    if args.mask_dir is None:
        print('ERROR: missing mask image dir ')
        return

    if args.xDim is None:
        print('ERROR: missing x-dimension; default = 512 ')
        xDim = 512
    else:
        try:
            xDim = int(args.xDim)
        except ValueError:
            print('ERROR: x-dimension is not a number; default = 512 ')
            xDim = 512

    if args.yDim is None:
        print('ERROR: missing y-dimension; default = 512 ')
        yDim = 512
    else:
        try:
            yDim = int(args.yDim)
        except ValueError:
            print('ERROR: y-dimension is not a number; default = 512 ')
            yDim = 512

    if xDim < 0 or yDim < 0:
        print('ERROR: xDim or yDim is less than zero')
        return

    tile_batch(args.image_dir, args.mask_dir, args.output_dir, xDim, yDim)

if __name__ == "__main__":
    main()