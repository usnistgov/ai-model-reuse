# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob
import os

''' 
this class is for computing statistics of GPU utilization from a CSV file
gathered by AI recommender
__author__      = "Peter Bajcsy"
__email__ = "peter.bajcsy@nist.gov"
'''



'''
This is the main routine for computing statistics:

average and stdev of GPU temp. [C]
average and stdev of GPU util. [%] over all epochs
average and stdev of GPU Memory used [MB]
Memory total [MB]

Note: Ideal GPU temperatures range from 65 to 85°C (149 to 185°F) under normal use condition
URL: https://www.avg.com/en/signal/check-gpu-temperature
As your GPU heats up, it can begin to self-regulate its performance to cool itself down

From URL: https://www.cgdirector.com/gpu-temperature-guide/
    Idle: 30° to 45° C (86° to 113° F)
    Load: 65° to 85° C (149° to 185° F)
    GPU Rendering: 70° to 80° C (158° F to 176° F)
    Gaming: 60° to 70° C (140° to 158° F)
    
Download (for AMD cards) MSI Afterburner https://www.msi.com/page/afterburner
Download (for Nvidia cards) EVGA Precision https://www.evga.com/precisionxoc/
You'll be able to manually increase the fan speed on you video card to help lower the temps.

'''
def compute_gpu_stats(in_path, out_path):

    # Check whether the specified output path exists or not
    doesExist = os.path.exists(out_path)
    if not doesExist:
        os.makedirs(out_path)

    # create header for the output metrics.csv file
    output_gpu_stats = out_path + os.path.sep + "gpu_stats.csv"
    print('INFO: output_gpu_stats:', output_gpu_stats)

    basename_label = 'CSV File Name'
    avg_temp_label = 'average of GPU temp. [C]'
    stdev_temp_label = 'stdev of GPU temp. [C]'
    avg_util_label = 'average of GPU util. [%]'
    stdev_util_label = 'stdev of GPU util. [%]'
    avg_mem_util_label = 'average of GPU Memory util. [%]'
    stdev_mem_util_label = 'stdev of GPU Memory util. [%]'
    total_mem_label = 'GPU Memory total [MB]'
    max_mem_util_label ='max of GPU Memory util. [%]'
    metrics_list = pd.DataFrame(columns=[basename_label, avg_temp_label, stdev_temp_label,
                                         avg_util_label, stdev_util_label, avg_mem_util_label, stdev_mem_util_label, total_mem_label, max_mem_util_label])
    metrics_list.to_csv(output_gpu_stats, mode='w', header=True, index=False)

    # csv_folder = os.path.abspath(in_path)
    # for csv_file in os.listdir(csv_folder):
    #     if csv_file.endswith(".csv"):
    #         print(os.path.join(in_path, csv_file))

    basename_metrics = []
    avg_temp = []
    stdev_temp = []
    avg_util = []
    stdev_util = []
    avg_mem_util = []
    stdev_mem_util = []
    total_mem = []
    max_mem_util = []

    os.chdir(in_path)
    for fname in glob.glob("*.csv"):
        print('INFO: fname:', fname)
        basename = os.path.basename(fname)
        print('INFO:: basename:', basename)


        if basename.endswith('gpu_stats.csv'):
            # skip the summary file named gpu_stats.csv
            continue

        ###########################################################
        # Note: to merge metrics.csv anf gpu_stats.csv tables, the names of csv files must be the same
        # input GPU files are "deeplab50_metrics_A10_1_gpu.csv" while metrics files are "deeplab50_metrics_A10_1.csv"
        basename = basename[:-8] + '.csv'
        basename_metrics.append(basename)

        df = pd.read_csv(fname)
        # gather data from the CSV file
        x_final = np.array(df['Epoch'][0:])
        gpu_temp = np.array(df['GPU temp. [C]'][0:])
        gpu_util = np.array(df['GPU util. [%]'][0:])
        mem_util = np.array(df['Memory util. [%]'][0:])
        mem_total = np.array(df['Memory total [MB]'][0:])

        ########################################
        # statistics
        # 1. avg and stdev of gpu temperature
        val = np.mean(gpu_temp)
        avg_temp.append(val)
        print('INFO: avg_temp:', val)

        val = np.std(gpu_temp)
        stdev_temp.append(val)
        print('INFO: stdev_temp:', val)
        #####################
        # 2. avg and stdev of gpu utilization
        val = np.mean(gpu_util)
        avg_util.append(val)
        print('INFO: avg_util:', val)

        val = np.std(gpu_util)
        stdev_util.append(val)
        print('INFO: stdev_util:', val)
        #####################
        # 3. avg and stdev of gpu mem utilization
        val = np.mean(mem_util)
        avg_mem_util.append(val)
        print('INFO: avg_mem_util:', val)

        val = np.std(mem_util)
        stdev_mem_util.append(val)
        print('INFO: stdev_mem_util:', val)
        #####################
        # 4. avg of total gpu mem
        val = np.mean(mem_total)
        total_mem.append(val)
        print('INFO: total_mem:', val)
        #####################
        # 5. max gpu mem
        val = np.max(mem_util)
        max_mem_util.append(val)
        print('INFO: max_mem_util:', val)
        #####################


    metrics_list[basename_label] = basename_metrics
    metrics_list[avg_temp_label] = avg_temp
    metrics_list[stdev_temp_label] = stdev_temp

    metrics_list[avg_util_label] = avg_util
    metrics_list[stdev_util_label] = stdev_util

    metrics_list[avg_mem_util_label] = avg_mem_util
    metrics_list[stdev_mem_util_label] = stdev_mem_util

    metrics_list[total_mem_label] = total_mem
    metrics_list[max_mem_util_label] = max_mem_util

    ##########################################
    #metrics_list = pd.DataFrame({"metrics": [array_metrics]})
    print(metrics_list)
    #metrics_list.to_csv(output_metrics_file, index=False)
    metrics_list.to_csv(output_gpu_stats, mode='a', header=False, index=False)



def main():
    # Setup the Argument parsing
    parser = argparse.ArgumentParser(prog='gpu_stats', description='Script which computes statistics of GPU utilization for comparing AI architectures')
    parser.add_argument('--input_dir', dest='input_dir', type=str, help='Folder where all input CSV files are located (Required)', required=True)
    parser.add_argument('--output_dir', dest='output_dir', type=str, help='Folder where output metrics will be saved (Required)', required=True)

    args = parser.parse_args()

    if args.input_dir is None:
        print('ERROR: missing input  dir ')
        return

    if args.output_dir is None:
        print('ERROR: missing output dir ')
        return

    input_folder = args.input_dir
    output_folder = args.output_dir

    print('Arguments:')
    print('input folder = {}'.format(input_folder))
    print('output folder = {}'.format(output_folder))
    compute_gpu_stats(input_folder, output_folder)


if __name__ == "__main__":
    main()
