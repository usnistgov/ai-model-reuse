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

"""
This class will fuse all regions of interest (ROIs) with the background
The fusion method is based on computing mean and stdev outside of the ROIs
and using Gaussian(mean, stdev) generator to replace the intensities in ROIs
 __author__      = "Peter Bajcsy"
__email__ = "peter.bajcsy@nist.gov"

"""

'''
The method is used for determining the threshold for low quality annotations
that missed some obvious ROIs (should have been annotated but are not)

formula adapted from https://stats.stackexchange.com/questions/311592/how-to-find-the-point-where-two-normal-distributions-intersect

This approach also assumes that every mask image contains at least one segmentation object of interest
'''


def mid_point_gaussian(mean1, stdev1, mean2, stdev2):
    # midpoint = (-b+/-sqrt(b^2-4ac)/2a
    # see https://en.wikipedia.org/wiki/Quadratic_equation

    a = -1.0 / (stdev1 * stdev1) + 1.0 / (stdev2 * stdev2)
    b = 2 * (-mean2 / (stdev2 * stdev2) + mean1 / (stdev1 * stdev1))
    c = mean2 * mean2 / (stdev2 * stdev2) - mean1 * mean1 / (stdev1 * stdev1) + np.log(
        stdev2 * stdev2 / (stdev1 * stdev1))
    d = b * b - 4 * a * c
    if d < 0:
        print('ERROR: discriminant is < 0:', d)
        return -1.0

    d = np.sqrt(d)
    x1 = (-b + d) / (2 * a)
    x2 = (-b - d) / (2 * a)
    print('INFO: x1:', x1, ' x2:', x2)

    if mean1 < mean2 and x1 > mean1 and x1 < mean2:
        return x1
    if mean1 > mean2 and x1 > mean2 and x1 < mean1:
        return x1

    if mean1 < mean2 and x2 > mean1 and x2 < mean2:
        return x2
    if mean1 > mean2 and x2 > mean2 and x2 < mean1:
        return x2

    print('WARNING: the two roots are not in-between the two means: mean1:', mean1, ' mean2:', mean2)
    print('WARNING: returning the mean1 (assumed to be mu of BKG) =  {} '.format(mean1))
    return mean1

    # find the closest root to one of the mean values
    # if abs(x1 - mean1) < abs(x1 - mean2):
    #     min_x1 = abs(x1 - mean1)
    # else:
    #     min_x1 = abs(x1 - mean2)
    # if abs(x2 - mean1) < abs(x2 - mean2):
    #     min_x2 = abs(x1 - mean1)
    # else:
    #     min_x2 = abs(x1 - mean2)
    # if min_x1 < min_x2:
    #     print('WARNING: returning the closest root x1= {} to one of the means'.format(x1))
    #     return x1
    # else:
    #     print('WARNING: returning the closest root x2= {} to one of the means'.format(x2))
    #     return x2


'''
This method decides on an intensity  threshold that separates Gaussian models of FRG and BKG
Then uses the BKG model for generating pixels inside of each FRG region
'''


def fuse_rois(image_folder, mask_folder, output_image_folder):
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
        basename = fn.rsplit('.', 1)  # split on the last occurrence of the delimiter

        if os.path.isfile(image_file) and os.path.isfile(mask_file):
            start = time.time()
            image = skimage.io.imread(fname=image_file)
            mask = skimage.io.imread(fname=mask_file)
            print('image.shape:', image.shape)
            print('mask.shape:', mask.shape)

            # estimate stats of image
            min_value = None
            max_value = None
            rows, cols = (image.shape[0], image.shape[1])
            sum_bkg, sum_frg = float(0.0), float(0.0)
            sum2_bkg, sum2_frg = float(0.0), float(0.0)
            count_bkg, count_frg = 0, 0
            for i in range(0, cols):
                for j in range(0, rows):
                    if mask[j][i] == 0:
                        sum_bkg += float(image[j][i])
                        sum2_bkg += float(image[j][i]) * float(image[j][i])
                        count_bkg += 1
                        if (min_value is None or image[j][i] < min_value):
                            min_value = image[j][i]
                        if (max_value is None or image[j][i] > max_value):
                            max_value = image[j][i]
                    else:
                        sum_frg += float(image[j][i])
                        sum2_frg += float(image[j][i]) * float(image[j][i])
                        count_frg += 1

            # derive average and stdev for FRG and BKG pixels
            mu_frg = float(sum_frg / count_frg)
            sigma_frg = float(sum2_frg / count_frg)
            if sigma_frg < mu_frg * mu_frg:
                print('ERROR: something is wrong with FRG stdev computation for image:', image_file)
                continue
            sigma_frg = np.sqrt(sigma_frg - mu_frg * mu_frg)
            ##########
            mu_bkg = float(sum_bkg / count_bkg)
            sigma_bkg = float(sum2_bkg / count_bkg)
            if sigma_bkg < mu_bkg * mu_bkg:
                print('ERROR: something is wrong with BKG stdev computation for image:', image_file)
                continue
            sigma_bkg = np.sqrt(sigma_bkg - mu_bkg * mu_bkg)

            print('INFO: benchmark for estimating FRG and BKG mu and stdev {:.2f}[s]'.format(time.time() - start))

            # tunn 2D image into 1D array
            temp = np.array(image)
            measured_bkg_s = temp.flatten()
            # clear the plot and save the histogram of FRG and BKG curves
            count_bins, bins, ignored = plt.hist(measured_bkg_s, 30, density=True)
            plt.clf()
            plt.title('Hist of background (red) and foreground (blue)')
            plt.plot(bins, 1 / (sigma_bkg * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu_bkg) ** 2 / (2 * sigma_bkg ** 2)),
                     linewidth=2, color='r')
            plt.plot(bins, 1 / (sigma_frg * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu_frg) ** 2 / (2 * sigma_frg ** 2)),
                     linewidth=2, color='b')
            hist_basename = 'hist_frg_bkg_' + basename[0] + '.png'
            out_hist_file = os.path.join(output_image_folder, hist_basename)
            plt.savefig(out_hist_file)
            plt.close()

            # decide on threshold for valid background intensities and remove values above the thresh
            thresh_background = mid_point_gaussian(mu_bkg, sigma_bkg, mu_frg, sigma_frg)
            print('INFO: mid point Gaussian threshold={}'.format(thresh_background))

            # this is fast and working implementation
            start = time.time()
            array_len = len(measured_bkg_s)
            print('DEBUG: before array_len:', array_len)
            if mu_bkg > mu_frg:
                print('WARNING: mean of FRG= {} is less than mean of BKG {}'.format(mu_frg, mu_bkg))
                threshold_indices = measured_bkg_s > thresh_background
            else:
                threshold_indices = measured_bkg_s < thresh_background

            measured_bkg_s = measured_bkg_s[threshold_indices]
            print('INFO: benchmark for thresh and delete elements {:.2f}[s]'.format(time.time() - start))
            array_len = len(measured_bkg_s)
            print('DEBUG: after array_len:', array_len)

            plt.clf()
            a = np.arange(int(thresh_background))  # Return evenly spaced values within a given interval.
            # hist, bin_edges = np.histogram(measured_s, bins=a)
            # plt.plot(bin_edges, hist, linewidth=2, color='r')
            plt.hist(measured_bkg_s, bins=a)
            plt.title('Hist of measured background ')
            # thresh:', str(int(thresh_background)),  '\n mu_bkg:', mu_bkg, ' sigma_bkg:', sigma_bkg, ' mu_frg:', mu_frg, ' sigma_frg:', sigma_frg )
            hist_basename = 'hist_meas_' + basename[0] + '.png'
            out_hist_file = os.path.join(output_image_folder, hist_basename)
            plt.savefig(out_hist_file)
            plt.close()

            # generate count points from the Gaussian PDF
            mu = np.mean(measured_bkg_s)
            sigma = np.std(measured_bkg_s)
            s = np.random.normal(mu, sigma, len(measured_bkg_s))

            # clear the plot and save the histogram
            count_bins, bins, ignored = plt.hist(s, 30, density=True)
            plt.clf()
            plt.title('Hist of background Gaussian PDF')
            plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
                     linewidth=2, color='r')
            hist_basename = 'hist_gen_' + basename[0] + '.png'
            out_hist_file = os.path.join(output_image_folder, hist_basename)
            plt.savefig(out_hist_file)
            plt.close()

            # clip the generated numbers
            s = np.clip(s, a_min=min_value, a_max=max_value)
            # save histogram after clipping the min and max values
            count_bins, bins, ignored = plt.hist(s, 30, density=True)
            plt.clf()
            plt.title('Hist of background Gaussian PDF after clipping min and max values')
            plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
                     linewidth=2, color='r')
            hist_basename = 'hist_gen2_' + basename[0] + '.png'
            out_hist_file = os.path.join(output_image_folder, hist_basename)
            plt.savefig(out_hist_file)
            plt.close()

            idx = 0
            for i in range(0, cols):
                for j in range(0, rows):
                    if mask[j][i] > 0:
                        image[j][i] = s[idx]
                        idx += 1

            # result_image = deepcopy(image)
            # result_mask = [[0 for i in range(cols)] for j in range(rows)]
            # result_mask = np.array(result_mask, dtype=np.uint8)
            skimage.io.imsave(fname=out_image_file, arr=image)


def main():
    parser = argparse.ArgumentParser(prog='fuse_rois', description='Script that fuses the ROIs with BKG')
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

    fuse_rois(args.image_dir, args.mask_dir, args.output_dir)


if __name__ == "__main__":
    main()
