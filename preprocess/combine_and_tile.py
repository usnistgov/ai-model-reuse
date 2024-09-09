# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.
import itertools
from PIL import Image
import os
import numpy as np
import argparse
from skimage.io import imread, imsave

"""
This class will combine image sequence and tile each image in the input folder into tiles saved in the output folder.

The arguments refer to the number of cuts along x- and y-axes (xPieces and yPieces). 
Thus, an input image of size 1024 x 1024 will be cut into tiles of size 512 x 512 with arguments
--xPieces 2 --yPieces 2

__author__  = "Pushkar Sathe"
__email__   = "pushkar.sathe@nist.gov"
"""


def parse_INFER_folder_names(folder_name):
    """
    Each folder is expected to be a set of 2D images corresponding to 1 4D or 5D measurement

    Folders are named by"
     "Experiment_identifier" _ "Sampletype" _ " AI Experiment Identifier" _ sampleid
    """
    try:
        expt_name, sample_type, ai_expt, sample_id = folder_name.split("_")
    except:
        expt_name = folder_name
        sample_type, ai_expt, sample_id = None, None, None

    return expt_name, sample_type, ai_expt, sample_id


def count_int_digits(n):
    """
    Given an integer, returns the number of digits in the number.
    If a non integer is provided, it is automatically converted to an integer.
    """
    count = len(str(int(n)))
    return count


def tile(final_stack, savepath, xPieces, yPieces, ext=".tif"):
    """
    Takes the final combined stack as input and tiles it along x and y axis into
    xPieces and yPieces respectively.
    """
    fs = final_stack.shape
    imgheight, imgwidth = fs[1], fs[2]
    height = imgheight // xPieces
    width = imgwidth // yPieces
    # Final tiles ignored as their larger sizes cant currently be handled.
    for i, j in itertools.product(range(yPieces), range(xPieces)):
        starty = j * height
        startx = i * width
        endy = starty + height
        endx = startx + width
        crop_stack = final_stack[..., startx:endx, starty:endy]
        imsave(f"{savepath}_{i}-{j}.{ext}", crop_stack)


def _get_image_from_path(image_path, astype):
    """
    Get a single image from a single path. Return it as a specified datatype
    """
    _img = imread(image_path)
    if astype is not None:
        _img = _img.astype(astype)
    return _img


def return_images_from_paths(image_paths, astype=None):
    """
    Returns a single image if single path is provided. If a list of paths is provided,
    then returns a list of stacked images
    :param astype: convert image format. (Not recommended as this can cause many errors down the line)
    :param image_paths: list of image paths or a single path
    :return:
    """

    if isinstance(image_paths, str):
        return _get_image_from_path(image_paths)
    else:
        assert isinstance(image_paths, (np.ndarray, list))
        image_list = []
        for image_path in image_paths:
            img = _get_image_from_path(image_path, astype)
            image_list.append(img)
        image_list = np.asarray(image_list)
        return image_list


def parse_INFER_file_name(name):
    """
    File names refer to file names from Measured or Genarated Data.

    NOTE: Due to discrepancies in Physics based simulation values,
    those files are not currently considered.

    :param name:
    :return:
    """

    moire, wavelength, z, xi, chext = name.split("_")
    ch, ext = os.path.splitext(chext)
    if ch.endswith('.ome'):  # handle .ome.tif
        ch = ch.replace('.ome', '')
        ext = 'ome' + ext
    else:
        ext = ext[1:]

    return moire, wavelength, z, float(xi), ch, ext


def combine(image_dir, output_dir, xPieces, yPieces, zPieces=None, imaging_modes=None, tomo=False):
    """
    Combines image sequences into stacks of dimensions (H,Ξ,X,Y) or (H,Ξ,X,Y,Z)
    """
    print(f"Combining {image_dir} into {output_dir}, xPieces: {xPieces}, yPieces:{yPieces}."
          f" Imaging Modes: {imaging_modes}")
    file_array, filename_array, xis, chs = [], [], [], []
    ext = None
    filenames = [f for f in os.listdir(image_dir) if not os.path.isdir(f) if f.__contains__(".tif")]
    for filename in filenames:
        filepath = os.path.join(image_dir, filename)
        file_array.append(filepath)
        filename_array.append(filename)
        _, _, _, xi, ch, ext = parse_INFER_file_name(filename)
        xis.append(xi), chs.append(ch)
    xiss = sorted(list(set(xis)))
    chss = sorted(list(set(chs)))

    if imaging_modes is None:
        imaging_modes = ['H0', 'H1', 'H1dark']
    else:
        print(imaging_modes, type(imaging_modes))
        if isinstance(imaging_modes, list):
            imaging_modes = [str(s) for s in imaging_modes]
            imaging_modes = [s.replace('Hdark', 'H1dark') for s in imaging_modes]
            imaging_modes = [s.replace('DF', 'H1dark') for s in imaging_modes]
            if len(imaging_modes) == 1:
                imaging_modes = imaging_modes[0].split(" ")
        # elif isinstance(imaging_modes, str):
    # print(imaging_modes, ch in chss, ch in imaging_modes)

    assert all(ch in chss for ch in imaging_modes), f"Channel must be a list of one" \
                                                    f" or more of 'H0', 'H1', 'H1dark', currently {imaging_modes}"

    image_dirname = os.path.basename(os.path.normpath(image_dir))
    expt_name, sample_type, ai_expt, sample_id = parse_INFER_folder_names(image_dirname)
    print(expt_name, sample_type, ai_expt, sample_id)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # image_stack combines all images (all xi and channels) into a single stack
    image_stack = return_images_from_paths(file_array)
    # print("image_stack", image_stack.dtype)
    # print(imaging_modes, ch in chs, ch in imaging_modes)
    assert all(ch in chs for ch in
               imaging_modes), f"Channel must be a list of one or more of 'H0', 'H1', 'H1dark', currently{imaging_modes}"
    nxis, nchs = list(np.unique(xiss)), list(np.unique(imaging_modes))  # sorted values
    if tomo:
        # For 3D, the value will be -3
        selected_stack = np.zeros((len(nxis), len(nchs)) + image_stack.shape[-3:], dtype=image_stack.dtype)
    else:
        selected_stack = np.zeros((len(nxis), len(nchs)) + image_stack.shape[-2:], dtype=image_stack.dtype)

    for f, filename in enumerate(os.listdir(image_dir)):
        if chs[f] in imaging_modes:
            selected_stack[nxis.index(xis[f]), nchs.index(chs[f])] = image_stack[f]
    ss_shape = selected_stack.shape
    # Combine XI and C dimensions
    final_stack = selected_stack.reshape((ss_shape[0] * ss_shape[1],) + ss_shape[2:])
    # add folder name
    if None in [sample_type, ai_expt, sample_id]:
        savepath = output_dir + "/" + f"{expt_name}"
    else:
        savepath = output_dir + "/" + f"{expt_name}_{sample_type}_{ai_expt}_{sample_id}"
    # print(ss_shape,"fsdtype", final_stack.dtype)
    print("EXT", ext)
    tile(final_stack, savepath, xPieces, yPieces, ext=ext)


def combine_subfolders(image_dir, output_dir, xPieces, yPieces, zPieces=None, usechannels=None):
    """
    It is assumed that the image_dir has a list of folders with names corresponding to mask names.
    Each folder contains a set of images that can be combined to obtain a single measurement.
    """
    print(f"Combining {image_dir} into {output_dir}, xPieces: {xPieces}, yPieces:{yPieces}. usechannels: {usechannels}")
    subdirs = False
    for name in os.listdir(image_dir):
        dirorfile = image_dir + "/" + name
        # print(dirorfile)
        if os.path.isdir(dirorfile):  # subdir
            combine(dirorfile, output_dir, xPieces, yPieces, imaging_modes=usechannels)
            subdirs = True
    if not subdirs:
        combine(image_dir, output_dir, xPieces, yPieces, imaging_modes=usechannels)


def main():
    parser = argparse.ArgumentParser(prog='combine_tile', description='Script that combines and tiles data')
    parser.add_argument('-i', '--image_dir', type=str, help='folder path to input images')
    parser.add_argument('-o', '--output_dir', type=str, help='folder path to saving output tiles')
    parser.add_argument('-c', '--channels', type=str, nargs='+', help='example input: ["H1dark"]')
    parser.add_argument('-x', '--xPieces', type=int, help='number of image cuts along x-axis')
    parser.add_argument('-y', '--yPieces', type=int, help='number of image cuts along y-axis')
    parser.add_argument('-z', '--zPieces', type=int, help='number of image cuts along y-axis')
    args, unknown = parser.parse_known_args()
    print("CHANNELS\t", args.channels)
    if args.image_dir is None:
        print('ERROR: missing input image dir ')
        return

    assert os.path.isdir(args.image_dir), "Image dir must be a directory"
    combine_subfolders(args.image_dir, args.output_dir, args.xPieces, args.yPieces, usechannels=args.channels,
                       zPieces=args.zPieces)


if __name__ == "__main__":
    main()
