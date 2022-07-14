# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import sys
import numpy as np
import skimage.color
import skimage.io
from matplotlib import pyplot as plt
from PIL import Image
import os
import argparse
import csv

"""
This class will generate a grayscale histogram plot for an image and save histogram plots and porosity summary CSV file 
  in the output folder.
  __author__      = "Peter Bajcsy"
__email__ = "peter.bajcsy@nist.gov"
"""


def class_histogram(image_dir, output_dir, max_num_labels, save_hist_flag):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # create the header for the resulting file
    header = ['filename', 'porosity']
    for i in range(0, max_num_labels):
        label = 'label ' + str(i)
        header.append(label)

    print('DEBUG: header:', header)

    # save the header
    out_histfile = output_dir + os.path.sep + 'summary_hist.csv'
    print('save hist file: ', out_histfile)
    with open(out_histfile, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)

    file_extension = '.tif'
    for filename in os.listdir(image_dir):
        filepath = os.path.join(image_dir, filename)
        print('filename:', filename)
        if not os.path.isfile(filepath):
            print('INFO: skip directory:',filename)
            continue

        if not filename.endswith(file_extension):
            print('INFO: skip filename other than ', file_extension, ' such as:', filename)
            continue

        image = skimage.io.imread(fname=filepath, as_gray=True)
        # create the histogram
        histogram, bin_edges = np.histogram(image, bins=max_num_labels, range=(0, max_num_labels))

        sum_material = 0 # material inside of the cylinder/container (excludes BKG=0, container wall=2
        for i in range(0,len(histogram)):
            #print('bin[', i, ']=', histogram[i])
            if i == 1 or (i > 2 and i < len(histogram)):
                sum_material += histogram[i]

        # compute porosity assuming that label 1 is the air label in the simulated masks
        # and label 2 is the container wall
        porosity = histogram[1]/sum_material
        porosity = int(porosity*1000)/1000

        with open(out_histfile, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # write the data
            data = [filename, porosity]
            for i in range(0, len(histogram)):
                data.append(histogram[i])
            # write one row per image
            writer.writerow(data)


        if save_hist_flag:
            # configure and draw the histogram figure
            plt.figure()
            plt.title("histogram\n" + filename)
            plt.xlabel("grayscale value")
            plt.ylabel("frequency of occurrence")
            #plt.xlim([0.0, 1.0])  # <- named arguments do not work here

            #plt.plot(bin_edges[0:-1], histogram)  # <- or here
            plt.bar(bin_edges[0:-1], histogram)
            #plt.show()
            outGraph_hist = output_dir + os.path.sep + filename
            outGraph_hist = outGraph_hist[:-len(file_extension)] + '_hist.png'
            print('INFO: outGraph_hist:', outGraph_hist)
            plt.tight_layout()
            plt.savefig(outGraph_hist)

            # clear the plot
            plt.clf()
            plt.close()




def main():
    parser = argparse.ArgumentParser(prog='class_histogram', description='Script that computes class histograms')
    parser.add_argument('--image_dir', type=str, required=True, dest='image_dir', default=None, help='folder path to input images')
    parser.add_argument('--max_labels', type=int, required=True, default=None, help='maximum number of labels')
    parser.add_argument('--output_dir', type=str, required=True, default=None, help='folder path to saving output histograms and porosity values')
    parser.add_argument('--save_hist', type=str, required=False, default="False", help='flag whether the histogram png files should be saved on disk')
    args, unknown = parser.parse_known_args()

    if args.image_dir is None:
        print('ERROR: missing input image dir ')
        return

    if args.max_labels is None:
        print('ERROR: missing max_labels ')
        return

    if args.output_dir is None:
        print('ERROR: missing output image dir ')
        return

    save_hist_flag = True
    if args.save_hist == "False":
        save_hist_flag = False

    max_num_labels = args.max_labels # 9 for infer

    class_histogram(args.image_dir, args.output_dir, max_num_labels, save_hist_flag)


if __name__ == "__main__":
    main()


