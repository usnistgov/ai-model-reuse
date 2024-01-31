# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.
from PIL import Image
import os
import argparse
import itertools
import re
import numpy as np
import pandas as pd
import tifffile

"""
This class will stitch tiles in the input folder into a mosaic of tiles saved as one image in the output folder.

The stitching assumes that the tiles were created by the tiling.py code.

TODO: the tiling followed by stitching does not produce identical pixel sizes of the stiched images as the original images
This has to be fixed on the tiling side (tile size is a floor of dimensions (height = imgheight // yPieces) ).
and in the stitching code (tiles are assumed to have the same dimensions as the first tile in the collection)
 
 __author__  = "Pushkar Sathe"
 __email__   = "pushkar.sathe@nist.gov"
"""


def stitch_subfolders(image_dir, output_dir, tomo=None):
    # create the output directory if needed
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # print("stitch sub")
    if tomo is None:
        for dirname in os.listdir(image_dir):
            dirpath = os.path.join(image_dir, dirname)
            if not os.path.isdir(dirpath):
                if dirpath.__contains__(".tif"):
                    # if tomo is None:
                    #     stitch(image_dir, output_dir)
                    #     print("tomo not selected")
                    # else:
                    #     print("tomo selected")
                    #     stitch_tomo(image_dir, output_dir)
                # else:
                    continue

            model_image_dir = dirpath
            model_output_dir = os.path.join(output_dir, dirname)
            print('model_image_dir:', model_image_dir)
            print('model_output_dir:', model_output_dir)
            # if tomo is None:
            stitch(model_image_dir, model_output_dir)
            # else:
            #     print("tomo selected1", model_image_dir)
            #     stitch_tomo(model_image_dir, model_output_dir)
    else: # assume tomo doesnt use subfolders
        stitch_tomo(image_dir, output_dir)
            # print(tomo)


def stitch(image_dir, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    mosaic_image_array = []
    max_xTilePos = []
    max_yTilePos = []

    # example filename and its decomposition
    # filename: pred_DEC_SizeVariation_xi000.011853_lambda000.336736_H03-4.tif
    # xTilePos: 4  yTilePos: 3  img_name: pred_DEC_SizeVariation_xi000.011853_lambda000.336736_H

    # step 1: decompose file names into sets of tiles that belong to the same mosaic image
    # estimate the maximum number of horizontal and vertical tiles for each mosaic image
    xTilePos = -1
    yTilePos = -1
    xTilePoss = []
    yTilePoss = []
    file_extension = '.tif'
    for filename in os.listdir(image_dir):
        filepath = os.path.join(image_dir, filename)
        # print('filename:', filename)
        if not os.path.isfile(filepath):
            continue

        # input filename: pred_CG1D_PS_MLMeasured_1_0-0.tif

        basename, ext = os.path.splitext(filename)
        bn_u = basename.split("_")
        img_name, xypos = "_".join(bn_u[:-1]), bn_u[-1]
        yTilePosl, xTilePosl = xypos.split("-")
        xTilePos = int("".join(xTilePosl))
        yTilePos = int("".join(yTilePosl))
        xTilePoss.append(xTilePos)
        yTilePoss.append(yTilePos)
        # i = 0
        mosaic_image_array.append(img_name)
    # print(xTilePoss)
    # print(yTilePoss)
    max_xTilePos.append(max(xTilePoss))
    max_yTilePos.append(max(yTilePoss))

    # print('max_xTilePos:', max_xTilePos, ' max_yTilePos:', max_yTilePos, ' mosaic_image_array:', mosaic_image_array)
    mosaic_image_array = list(set(mosaic_image_array))  # list of all unique file names after stitching?
    print('max_xTilePos:', max_xTilePos, ' max_yTilePos:', max_yTilePos, ' mosaic_image_array:', mosaic_image_array)

    # step 2: place all tiles in one collection into its corresponding positions of a mosaic image
    # based on its tile indices
    k = 0
    for file in mosaic_image_array:
        result = None
        finalWidth = -1
        finalHeight = -1

        for i in range(0, int(max_yTilePos[k]) + 1):
            for j in range(0, int(max_xTilePos[k]) + 1):
                # this takes care of the old version of naming convention (see above)
                filename = file + "_" + str(i) + "-" + str(j) + file_extension

                filepath = os.path.join(image_dir, filename)
                # print('input filename:', filename)

                file_exists = os.path.exists(filepath)
                if not file_exists:
                    print('INFO: missing a tile in the grid: column=', j, ' row=', i)
                    continue
                img = Image.open(filepath)
                # numcols, numrows = img.size #FOR NORMAL?
                numrows, numcols = img.size  # FOR LSTM
                if finalWidth == -1 or finalHeight == -1:
                    # check the last tile for an odd size due to tiling algorithm
                    lasttile_filename = file + "_" + str(int(max_yTilePos[k])) + "-" + str(
                        int(max_xTilePos[k])) + file_extension
                    lasttile_filepath = os.path.join(image_dir, lasttile_filename)
                    print('lasttile_filepath:', lasttile_filepath)
                    img = Image.open(lasttile_filepath)
                    lasttile_numcols, lasttile_numrows = img.size
                    print("INFO: last tile dim: %s: %s, %s x %s" % (
                        filepath, img.mode, lasttile_numcols, lasttile_numrows))

                    # init the final mosaic image

                    print("INFO: regular tile dim: %s: %s, %s x %s" % (filepath, img.mode, numcols, numrows))

                    finalWidth = numcols * (int(max_xTilePos[k])) + lasttile_numcols
                    finalHeight = numrows * (int(max_yTilePos[k])) + lasttile_numrows

                    # see the modes at https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
                    # The mode= "I;16" did not work; The mode "L" is good for 8 bit masks
                    # set the output image mode to the same as the input tile image mode
                    result = Image.new(mode=img.mode, size=(finalWidth, finalHeight))
                    result.paste(img, (j * numcols, i * numrows))
                else:
                    img = Image.open(filepath)
                    result.paste(img, (j * numcols, i * numrows))
        if result is not None:
            # step 3: save the stitched image
            if file.endswith('-'):
                file = file[0:len(file) - 1]

            out_filename = file + file_extension
            out_filepath = os.path.join(output_dir, out_filename)
            print('save mosaic file: ', out_filepath)
            result.save(out_filepath)


def stitch_tomo(image_dir, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # example filename and its decomposition
    # filename: pred_DEC_SizeVariation_xi000.011853_lambda000.336736_H0_6-3-4.tif
    # zTilePos: 6 xTilePos: 4  yTilePos: 3  img_name: pred_DEC_SizeVariation_xi000.011853_lambda000.336736_H0

    # step 1: decompose file names into sets of tiles that belong to the same mosaic image
    # estimate the maximum number of horizontal and vertical tiles for each mosaic image
    file_extension = 'tiff'
    mosaic_image_array = []
    max_xTilePos = []
    max_yTilePos = []
    max_zTilePos = []
    mosaic_df = pd.DataFrame(columns=["mosaic_image_array", "z", "y", "x"])
    maxz, maxy, maxx = -1, -1, -1
    z_len, y_len, x_len = -1, -1, -1
    dims = -1
    xis = -1
    for f, filename in enumerate(os.listdir(image_dir)):
        filepath = os.path.join(image_dir, filename)
        # print('filename:', filename, end="\t")
        if not os.path.isfile(filepath):
            continue
        if f == 0:
            temp_img = tifffile.imread(filepath)
            dims = temp_img.ndim
            print(f"DIMS = {dims}, SHAPE = {temp_img.shape}")
            if dims == 3:
                z_len, y_len, x_len = temp_img.shape
            if dims == 4:
                xis, z_len, y_len, x_len = temp_img.shape
        # input filename: <name>-zPieces-yPieces-xPieces.suffix
        # input filename: pred_well1.hrf48.red_1-0.tif
        basename = re.split('_|\.', filename)
        img_name = "_".join(basename[:-2])
        # print("basename:", basename, end="\t")
        indices = basename[-2]
        z, y, x = indices.split('-')
        z, y, x = int(z), int(y), int(x)
        # print('zTilePos:', z, ' yTilePos:', y, 'xTilePos:', x, ' img_name:', img_name)
        maxz = z if z > maxz else maxz
        maxy = y if y > maxy else maxy
        maxx = x if x > maxx else maxx
        # mosaic_df.append({"mosaic_image_array": img_name, "z": z, "y": y, "x": x}, ignore_index=True)
        temp_df = pd.DataFrame([{"mosaic_image_array": img_name, "z": z, "y": y, "x": x}])
        mosaic_df = pd.concat([mosaic_df, temp_df], axis=0, ignore_index=True)
        # mosaic_image_array.append(img_name)
        # max_xTilePos.append(x)
    # print("MOSAIC READY", mosaic_df["mosaic_image_array"])
    # max_yTilePos.append(y)
    # max_zTilePos.append(z)
    for file in mosaic_df["mosaic_image_array"].unique():
        print("\nFILE:", file, z_len, x_len, y_len)
        # exit()
        if dims == 3:
            mosaic_image = np.zeros((z_len * (maxz + 1), y_len * (maxy + 1), x_len * (maxx + 1)), dtype=np.uint16)
            print(mosaic_image.dtype)
            for (k, i, j) in itertools.product(range(maxz + 1), range(maxy + 1), range(maxx + 1)):
                filepath = f"{file}_{k}-{i}-{j}.{file_extension}"
                image = tifffile.imread(os.path.join(image_dir, filepath))
                print(mosaic_image.shape, image.shape, i, j, k, x_len, y_len, z_len)
                mosaic_image[k * z_len:(k + 1) * z_len, i * y_len:(i + 1) * y_len, j * x_len:(j + 1) * x_len] = image
            print(mosaic_image.dtype)
            tifffile.imwrite(f"{os.path.join(output_dir, file)}.{file_extension}", data=mosaic_image, imagej=True)
        if dims == 4:
            mosaic_image = np.zeros((xis, z_len * (maxz + 1), y_len * (maxy + 1), x_len * (maxx + 1)), dtype=np.uint16)
            for (k, i, j) in itertools.product(range(maxz + 1), range(maxx + 1), range(maxy + 1)):
                filepath = f"{file}_{k}-{i}-{j}.{file_extension}"
                image = tifffile.imread(os.path.join(image_dir, filepath))
                print(mosaic_image.shape, image.shape, i, j, k, x_len, y_len, z_len)
                mosaic_image[:, k * z_len:(k + 1) * z_len, i * y_len:(i + 1) * y_len, j * x_len:(j + 1) * x_len] = image
            tifffile.imwrite(f"{os.path.join(output_dir, file)}.{file_extension}", data=mosaic_image, imagej=True)


def main():
    parser = argparse.ArgumentParser(prog='stitch', description='Script that stitches tiled images')
    parser.add_argument('--image_dir', type=str, help='folder path to a set of sub-folders with input tiled images')
    parser.add_argument('--output_dir', type=str, help='folder path to saving output stitched images')
    parser.add_argument('--tomo', type=str, nargs="+", help='folder path to saving output stitched images')

    # parser.add_argument('--xPieces', type=int, help='number of image cuts along x-axis')
    # parser.add_argument('--yPieces', type=int, help='number of image cuts along y-axis')
    args, unknown = parser.parse_known_args()
    if args.image_dir is None:
        print('ERROR: missing input image dir ')
        return

    if args.output_dir is None:
        print('ERROR: missing output image dir ')
        return
    # example of arguments:
    # --image_dir /home/pnb/trainingOutput/pytorchOutput_A10/infer_test_images/ --output_dir /home/pnb/trainingOutput/pytorchOutput_A10/infer_stitch/
    # --image_dir /home/pnb/trainingOutput/pytorchOutput_cryoem/infer_test_images/ --output_dir /home/pnb/trainingOutput/pytorchOutput_cryoem/infer_stitch/
    # stitch(args.image_dir, args.output_dir)
    print("TOMO:", args.tomo)
    stitch_subfolders(args.image_dir, args.output_dir, args.tomo)

    # stitch_subfolders(args.image_dir, args.output_dir)


if __name__ == "__main__":
    main()
