import argparse
import os
from fractions import Fraction
import math
from sklearn.model_selection import train_test_split
import shutil
import random

"""
This class will combine two datasets and mix them in the chosen ratio:

For example, 
If ratio = 30
Generated_DD                            Generated_PBS
    |-------->train_images                  |-------->train_images
    |-------->train_masks                   |-------->train_masks
    |-------->test_images                   |-------->test_images
    |-------->test_masks                    |-------->test_masks
    
The Results will be combned into a new folder 
Combined/3-7_PBS-DD
            |-------->train_images
            |-------->train_masks
            |-------->test_images
            |-------->test_masks

The folder name will attempt to use the simplest fraction under with denominator under 100, and above 10.

__author__      = "Pushkar Sathe"
__email__ = "pushkar.sathe@nist.gov"
"""


def percentage_to_simplest_fraction(percent):
    print(percent, type(percent))  # must be int or fraction. Floats arent always rational
    fraction = Fraction(numerator=int(percent), denominator=100).limit_denominator(100)
    simple_numerator = fraction.numerator
    simple_denominator = fraction.denominator
    if fraction.numerator + fraction.denominator < 10:
        x = fraction.numerator + fraction.denominator
        closest_multiple = math.ceil(10 / x)
        simple_numerator = fraction.numerator * closest_multiple
        simple_denominator = fraction.denominator * closest_multiple
        print("fraction:", x, closest_multiple, fraction)
    return f"{simple_numerator}-{simple_denominator - simple_numerator}"


def get_files(folder):
    """Retrieve file paths and names from a folder."""
    files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    filenames = [os.path.basename(f) for f in files]
    return files, filenames


def stratify_and_mix(src_folder1, src_folder2, trainortest, selected_perctentage, target_foldername, suffixes=None):
    # for suffix in suffixes:
    if suffixes is None:
        suffixes = ["masks", "images"]
    all_files = {}
    all_filenames = {}
    src_folders = [src_folder1, src_folder2]
    for i, src_folder in enumerate(src_folders):
        for suffix in suffixes:
            all_files[f"{i}_{trainortest}_{suffix}"], all_filenames[f"{i}_{trainortest}_{suffix}"] = get_files(
                os.path.join(src_folder, f"{trainortest}_{suffix}"))

            # Assuming files are paired based on sorting
            all_files[f"{i}_{trainortest}_{suffix}"].sort()
            all_filenames[f"{i}_{trainortest}_{suffix}"].sort()
    n_select_masks = []
    selected_masks = []
    n_select_images = []
    selected_images = []
    percentages = [selected_perctentage, 100 - selected_perctentage]
    for i, src_folder in enumerate(src_folders):
        # images and masks should have the same fiels
        assert all_filenames[f"{i}_{trainortest}_{suffixes[0]}"] == all_filenames[
            f"{i}_{trainortest}_{suffixes[1]}"], f"ERROR: {src_folder}, {i},{trainortest}"

        # Select percentage from first folder, and 100-percentage from second.
        total_files = len(all_filenames[f"{i}_{trainortest}_{suffixes[0]}"])
        # total_files = len(files[files[f"{0}_{traintest}_{suffixes[0]}"]]) + len(files[files[f"{1}_{traintest}_{suffix}"]])
        random.shuffle(all_filenames[f"{i}_{trainortest}_{suffixes[0]}"])
        print(f"Selecting {percentages[i]} percent from {src_folder}")
        usefraction = int(percentages[i] / 100 * total_files)
        print(percentages[i] / 100 * total_files, usefraction)
        n_select_masks = all_filenames[f"{i}_{trainortest}_{suffixes[0]}"][:usefraction]
        n_select_images = all_filenames[f"{i}_{trainortest}_{suffixes[0]}"][:usefraction]  # use same filenames
        src_mask_folder = os.path.join(src_folder, f"{trainortest}_masks")
        src_images_folder = os.path.join(src_folder, f"{trainortest}_images")
        selected_masks.extend(os.path.join(src_mask_folder, f) for f in n_select_masks)
        selected_images.extend(os.path.join(src_images_folder, f) for f in n_select_images)

    # n_select_from_folder1 = int(total_files * ratio / 100)
    assert len(selected_masks) == total_files, f"Total files {total_files}, {len(n_select_masks)}"
    assert len(selected_images) == total_files, f"Total files {total_files}, {len(n_select_images)}"

    combined_masks = os.path.join(target_foldername, f"{trainortest}_{suffixes[0]}")
    combined_images = os.path.join(target_foldername, f"{trainortest}_{suffixes[1]}")
    os.makedirs(combined_masks, exist_ok=True)
    os.makedirs(combined_images, exist_ok=True)
    selected_files = []
    for mfile_path in selected_masks:
        shutil.copy(mfile_path, combined_masks)
    for ifile_path in selected_images:
        shutil.copy(ifile_path, combined_images)


def mix_folders(src_folder1, src_folder2, ratio, dest_folder):
    """
    e.g.
    src_folder1 = ./Generated_PBS
    src_folder2 = ./Generated_DD
    ratio = 30
    dest_folder = ./Combined
    """
    # Names of subfolders
    # categories = ['train_images', 'train_masks', 'test_images', 'test_masks']
    train_test = ['train', 'test']
    # folder names should have _
    sf1_type = src_folder1.split("_")[-1]
    sf2_type = src_folder2.split("_")[-1]
    # TODO: add handling of naming for more options.
    rationame = f"{percentage_to_simplest_fraction(ratio)}_{sf1_type}-{sf2_type}"
    for train_or_test in train_test:
        combined_subfolder = os.path.join(dest_folder, rationame)
        os.makedirs(combined_subfolder, exist_ok=True)
        selected_files = stratify_and_mix(src_folder1, src_folder2, train_or_test, ratio, combined_subfolder)


def main():
    parser = argparse.ArgumentParser(description="Combine folder contents in a stratified and shuffled manner.")
    parser.add_argument('--src_folder1', type=str, required=True, help='Path to the first set of folders.')
    parser.add_argument('--src_folder2', type=str, required=True, help='Path to the second set of folders.')
    parser.add_argument('--ratio', type=float, required=True,
                        help='Ratio for combining folders in percent of folder 1. folder 2 will use 100-ratio.')
    parser.add_argument('--dest_folder', type=str, required=True, help='Destination folder path.')

    args = parser.parse_args()

    if not os.path.exists(args.dest_folder):
        os.makedirs(args.dest_folder, exist_ok=True)

    mix_folders(args.src_folder1, args.src_folder2, args.ratio, args.dest_folder)
    print(f"Folders have been combined based on a {args.ratio} ratio and saved to '{args.dest_folder}'.")


if __name__ == "__main__":
    main()
