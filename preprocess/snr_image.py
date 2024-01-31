# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import argparse
import sys
import time

import skimage.io
import pandas as pd
import numpy as np

"""
This class will compute signal to noise ratio (SNR) per image in the input folder
and save the values in a CSV file
__author__      = "Peter Bajcsy"
__email__ = "peter.bajcsy@nist.gov"
"""

'''
compute SNR for one image
'''


def snr_image(img, axis=0, ddof=0):
    img = np.asanyarray(img)
    m = img.mean(axis)
    sd = img.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)


'''
compute snr for all images in an image folder
'''


def folder_snr_image(image_folder, output_image_folder, dataset_name):
    image_folder = os.path.abspath(image_folder)
    image_files = [fn for fn in os.listdir(image_folder)]

    ###############################################
    out_file = dataset_name + '_snr_image.csv'
    out_csv_file = os.path.join(output_image_folder, out_file)
    dataset_label = 'Dataset label'
    basename_label = 'File name'
    snr_label = 'SNR'
    cv_label = 'CV'

    metrics_list = pd.DataFrame(columns=[dataset_label, basename_label, snr_label, cv_label])
    metrics_list.to_csv(out_csv_file, mode='w', header=True, index=False)
    #################################################
    start = time.time()

    dataset_arr = []
    basename_arr = []
    snr_arr = []
    cv_arr = []
    for fn in image_files:
        image_file = os.path.join(image_folder, fn)
        image = skimage.io.imread(fname=image_file)

        snr_val = snr_image(image, axis=None)
        print('INFO: image file name: ', fn, ' SNR: ', snr_val)
        cv_val = 1.0 / snr_val
        print('INFO: image file name: ', fn, ' CV: ', cv_val)
        dataset_arr.append(dataset_name)
        basename_arr.append(fn)
        snr_arr.append(snr_val)
        cv_arr.append(cv_val)

    # save the metrics per input file
    metrics_list[dataset_label] = dataset_arr
    metrics_list[basename_label] = basename_arr
    metrics_list[snr_label] = snr_arr
    metrics_list[cv_label] = cv_arr
    print(metrics_list)
    metrics_list.to_csv(out_csv_file, mode='a', header=False, index=False)
    ##########################################
    total_time = time.time() - start
    print('INFO: processing time [s]: ', total_time)
    return snr_arr, cv_arr


'''
compute snr for all images in image folder and the statistics over all images
'''


def batch_snr_image(image_folder, mask_folder, output_image_folder, dataset_name):
    # check that the output folder exists
    if not os.path.exists(output_image_folder):
        os.mkdir(output_image_folder)

    out_stats = 'snr_image_stats.csv'
    out_csv_stats = os.path.join(output_image_folder, out_stats)
    dataset_label = 'Dataset label'
    snr_avg_label = 'Avg. SNR'
    snr_stdev_label = 'Stdev. SNR'
    snr_min_label = 'Min SNR'
    snr_max_label = 'Max SNR'

    cv_avg_label = 'Avg. CV'
    cv_stdev_label = 'Stdev. CV'
    cv_min_label = 'Min CV'
    cv_max_label = 'Max CV'

    metrics_stats = pd.DataFrame(columns=[dataset_label, snr_avg_label, snr_stdev_label, snr_min_label, snr_max_label,
                                          cv_avg_label, cv_stdev_label, cv_min_label, cv_max_label])
    # create the output file only if it does not exist
    if not os.path.exists(out_csv_stats):
        metrics_stats.to_csv(out_csv_stats, mode='w', header=True, index=False)
    ##################################
    # TODO switch between snr types of computation
    # snr_arr, cv_arr = folder_snr_image(image_folder, output_image_folder, dataset_name)
    snr_arr, cv_arr = folder_snr_rois_image(image_folder, mask_folder, output_image_folder, dataset_name)

    ##########################################
    dataset_arr2 = []
    dataset_arr2.append(dataset_name)

    snr_avg, snr_stdev, snr_min, snr_max = [], [], [], []
    snr_avg.append(np.mean(snr_arr))
    snr_stdev.append(np.std(snr_arr))
    snr_min.append(np.min(snr_arr))
    snr_max.append(np.max(snr_arr))

    cv_avg, cv_stdev, cv_min, cv_max = [], [], [], []
    cv_avg.append(np.mean(cv_arr))
    cv_stdev.append(np.std(cv_arr))
    cv_min.append(np.min(cv_arr))
    cv_max.append(np.max(cv_arr))

    metrics_stats[dataset_label] = dataset_arr2
    metrics_stats[snr_avg_label] = snr_avg
    metrics_stats[snr_stdev_label] = snr_stdev
    metrics_stats[snr_min_label] = snr_min
    metrics_stats[snr_max_label] = snr_max

    metrics_stats[cv_avg_label] = cv_avg
    metrics_stats[cv_stdev_label] = cv_stdev
    metrics_stats[cv_min_label] = cv_min
    metrics_stats[cv_max_label] = cv_max

    print(metrics_stats)
    metrics_stats.to_csv(out_csv_stats, mode='a', header=False, index=False)


'''
compute snr for all images over a mask (0 - bkg, other - frg) in an image folder
'''


def folder_snr_rois_image(image_folder, mask_folder, output_image_folder, dataset_name):
    image_folder = os.path.abspath(image_folder)
    # image_files = [fn for fn in os.listdir(image_folder)]

    mask_folder = os.path.abspath(mask_folder)
    mask_files = [fn_mask for fn_mask in os.listdir(mask_folder)]
    ###############################################
    out_file = dataset_name + '_snr_rois_image.csv'
    out_csv_file = os.path.join(output_image_folder, out_file)
    dataset_label = 'Dataset label'
    basename_label = 'File name'
    snr_label = 'SNR'
    cv_label = 'CV'

    metrics_list = pd.DataFrame(columns=[dataset_label, basename_label, snr_label, cv_label])
    metrics_list.to_csv(out_csv_file, mode='w', header=True, index=False)

    start = time.time()
    dataset_arr = []
    basename_arr = []
    snr_arr = []
    cv_arr = []
    for fn_mask in mask_files:
        image_file = os.path.join(image_folder, fn_mask)
        mask_file = os.path.join(mask_folder, fn_mask)
        # basename = fn_mask.rsplit('.', 1)  # split on the last occurrence of the delimiter

        if os.path.isfile(image_file) and os.path.isfile(mask_file):
            image_file = os.path.join(image_folder, fn_mask)
            image = skimage.io.imread(fname=image_file)
            mask_file = os.path.join(mask_folder, fn_mask)
            mask = skimage.io.imread(fname=mask_file)

            # image = np.asarray(Image.open(mask_file))
            if mask.shape[0] != image.shape[0] or mask.shape[1] != image.shape[1]:
                print('ERROR: mismatch of mask and image sizes')
                print('image size:', image.shape[0], ',', image.shape[1])
                print('mask size:', mask.shape[0], ',', mask.shape[1])
                continue

            sum_signal, count_signal = 0.0, 0
            sum_noise, sum2_noise, count_noise = 0.0, 0.0, 0
            for h in range(mask.shape[0]):
                for w in range(mask.shape[1]):
                    mask_value = mask[h][w]
                    if mask_value > 0:
                        sum_signal += np.float64(image[h][w])
                        count_signal += 1
                    else:
                        sum_noise += image[h][w]
                        sum2_noise += np.float64(image[h][w] * image[h][w])
                        count_noise += 1

            if count_signal > 0:
                mean_signal = sum_signal / count_signal
            else:
                mean_signal = 0.0

            if count_noise > 0:
                sum_noise = sum_noise / count_noise
                stdev_noise = np.sqrt(sum2_noise / count_noise + sum_noise * sum_noise)
            else:
                stdev_noise = 0.0

            if stdev_noise > 0.0:
                snr_val = mean_signal / stdev_noise
                cv_val = 1.0 / snr_val
            else:
                snr_val = 0.0
                cv_val = sys.float_info.max

            print('INFO: image file name: ', fn_mask, ' SNR: ', snr_val)
            print('INFO: image file name: ', fn_mask, ' CV: ', cv_val)
            dataset_arr.append(dataset_name)
            basename_arr.append(fn_mask)
            snr_arr.append(snr_val)
            cv_arr.append(cv_val)

    # save the metrics per input file
    metrics_list[dataset_label] = dataset_arr
    metrics_list[basename_label] = basename_arr
    metrics_list[snr_label] = snr_arr
    metrics_list[cv_label] = cv_arr
    print(metrics_list)
    metrics_list.to_csv(out_csv_file, mode='a', header=False, index=False)

    total_time = time.time() - start
    print('INFO: processing time [s]: ', total_time)
    return snr_arr, cv_arr


def main():
    parser = argparse.ArgumentParser(prog='SNR', description='Script that computes SNR value per image and per ROI')
    parser.add_argument('--image_dir', type=str, help='full path of image folder')
    parser.add_argument('--mask_dir', type=str, help='full path of mask folder')
    parser.add_argument('--output_dir', type=str, help='full path to output folder destination')
    parser.add_argument('--name_dataset', type=str, help='subfolder name according to the dataset')

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
    if args.name_dataset is None:
        print('ERROR: missing name_dataset ')
        return

    dataset_label = args.name_dataset
    # batch_snr_image(args.image_dir, args.output_dir, dataset_label)
    batch_snr_image(args.image_dir, args.mask_dir, args.output_dir, dataset_label)


if __name__ == "__main__":
    main()
