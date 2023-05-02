# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import argparse
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob
import os

''' 
this class is for parsing the data gathered by AI recommender
and plotting the train and test loss over time
as well as fitting a power function f(x) = a * x^b and the exponential function f(x) = a * b^x to the first N points
__author__      = "Peter Bajcsy"
__email__ = "peter.bajcsy@nist.gov"
'''


def power_func(x, a, b):
    return a * np.power(x, b)


def exp_func(x, a, b):
    return a * np.power(b, x)


def linear_func(x, a, b):
    return a * x + b


'''
This method estimates power function and exponential function coefficients from the first N points
the estimated values and their delta with respect to measured values are returned (and added to the Panda Data Frame)
The coefficients are saved to output_coefficients_file
'''


def estimate_pred_funcion(x_points, y_points, x_final, y_final_loss, df, basename, label_name, output_coeficients_file):
    plt.scatter(x_points, y_points)

    # estimate power function and exponential function from the first N points
    guess = [1.0, 1.0]
    pow_flag = True
    try:
        power_popt, power_pcov = curve_fit(power_func, x_points, y_points, p0=guess, maxfev=1000)
    except:
        print('ERROR: could not find a power function fit; use linear fit')
        power_popt, power_pcov = curve_fit(linear_func, x_points, y_points, maxfev=1000)
        pow_flag = False

    exp_flag = True
    try:
        exp_popt, exp_pcov = curve_fit(exp_func, x_points, y_points, maxfev=1000)
    except:
        print('ERROR: could not find an exponential function fit; use linear fit')
        exp_popt, exp_pcov = curve_fit(linear_func, x_points, y_points, maxfev=1000)
        exp_flag = False

    power_y_est_points = []
    power_delta_points = []
    exp_y_est_points = []
    exp_delta_points = []
    # cap the model-based prediction values to this cap_value
    cap_value = float(1e50)
    for i in range(0, len(x_final)):
        if pow_flag:
            val = power_func(x_final[i], power_popt[0], power_popt[1])
        else:
            val = linear_func(x_final[i], power_popt[0], power_popt[1])

        if math.isinf(val) or isinstance(val, type(None)) or math.fabs(val) > cap_value:
            power_y_est_points.append(cap_value)
        else:
            power_y_est_points.append(val)

        power_delta_points.append(power_y_est_points[i] - y_final_loss[i])

        if exp_flag:
            val = exp_func(x_final[i], exp_popt[0], exp_popt[1])
        else:
            val = linear_func(x_final[i], exp_popt[0], exp_popt[1])

        if math.isinf(val) or isinstance(val, type(None)) or math.fabs(val) > cap_value:
            exp_y_est_points.append(cap_value)
        else:
            exp_y_est_points.append(val)

        exp_delta_points.append(exp_y_est_points[i] - y_final_loss[i])

    est_points_label = 'PowerF_' + label_name
    df[est_points_label] = power_y_est_points
    delta_label = 'Delta_' + est_points_label
    df[delta_label] = power_delta_points

    est_points_label = 'ExpF_' + label_name
    df[est_points_label] = exp_y_est_points
    delta_label = 'Delta_' + est_points_label
    df[delta_label] = exp_delta_points

    coeff_points = pd.DataFrame([[basename, label_name, power_popt[0], power_popt[1], exp_popt[0], exp_popt[1]]])
    coeff_points.to_csv(output_coeficients_file, mode='a', header=False, index=False)
    return power_y_est_points, power_delta_points, exp_y_est_points, exp_delta_points, pow_flag, exp_flag


'''
This is the main routine for creating graphs and computing predictions
'''


def create_graphs(in_path, out_path):
    # in_path = "/home/pnb/trainingOutput/pytorchOutput_cryoem/*.csv"
    # out_path = "/home/pnb/trainingOutput/pytorchOutput_cryoem/graphs"

    # Check whether the specified output path exists or not
    doesExist = os.path.exists(out_path)
    if not doesExist:
        os.makedirs(out_path)

    output_coeficients_file = out_path + os.path.sep + "coefficients.csv"
    print('INFO: output_coeficients_file:', output_coeficients_file)
    coeff_points = pd.DataFrame(
        columns=['CSV File Name', 'Point Selection', 'power multiplier', 'power exponent', 'exp multiplier',
                 'exp base'])
    coeff_points.to_csv(output_coeficients_file, mode='a', header=True, index=False)

    # csv_folder = os.path.abspath(in_path)
    # for csv_file in os.listdir(csv_folder):
    #     if csv_file.endswith(".csv"):
    #         print(os.path.join(in_path, csv_file))

    os.chdir(in_path)
    for fname in glob.glob("*.csv"):
        print('INFO: fname:', fname)
        basename = os.path.basename(fname)
        print('INFO:: basename:', basename)

        if basename.endswith('coefficients.csv'):
            # skip the summary file named coefficients.csv
            continue

        df = pd.read_csv(fname, engine='python', warn_bad_lines=True, error_bad_lines=True)

        if df.shape[0] < 20:
            print('ERROR: reading file:', fname)
            print('ERROR - insufficient number of epochs < 20: df:', df.shape[0])
            print('DEBUG: df=', df)
            continue

        # gather data for power function fit from the CSV file

        x_final_time = np.array(df['Seconds'][0:])
        x_final = np.array(df['epoch'][0:])
        train_loss = np.array(df['Train_loss'][0:])
        test_loss = np.array(df['Test_loss'][0:])

        ## save teh Panda DataFrame with all added columns to the file outCSVupdate
        outCSVupdate = out_path + os.path.sep + basename
        print('INFO: save updated CSV file:', outCSVupdate)
        df.to_csv(outCSVupdate, index=False)

        # clear the plot
        plt.clf()
        plt.close()

        fig1 = plt.figure(1)
        plt.scatter(x_final, np.array(df['Train_loss'][0:]), c='green', s=10, label='Train Loss')
        plt.scatter(x_final, np.array(df['Test_loss'][0:]), c='blue', s=10, label='Test Loss')

        plt.legend(loc="upper right")
        title_train = basename[:-4] + ': Loss = f(epoch index)'
        plt.title(title_train)
        plt.xlabel('epoch index')
        plt.ylabel('Cross Entropy Loss')
        outGraph_train = out_path + os.path.sep + basename
        outGraph_train = outGraph_train[:-4] + '_loss.png'
        print('INFO: outGraph_train:', outGraph_train)
        plt.savefig(outGraph_train)

        # clear the plot
        plt.clf()
        plt.close()

        fig2 = plt.figure(2)
        plt.scatter(x_final, np.array(df['Dice'][0:]), c='red', s=10, label='Dice Coefficient')
        plt.scatter(x_final, np.array(df['Jaccard'][0:]), c='green', s=10, label='IoU score')
        plt.scatter(x_final, np.array(df['F1-Score'][0:]), c='blue', s=10, label='F-1 score')
        # plt.ylim([25, 50])

        plt.legend(loc="upper right")
        title_test = basename[:-4] + ': Test Loss = f(epoch index)'
        plt.title(title_test)
        plt.xlabel('epoch index')
        plt.ylabel('Metric score')
        outGraph_test = out_path + os.path.sep + basename
        outGraph_test = outGraph_test[:-4] + '_test.png'
        print('INFO: outGraph_test:', outGraph_test)
        plt.savefig(outGraph_test)

    # plt.show()


def main():
    # Setup the Argument parsing
    parser = argparse.ArgumentParser(prog='create_graphs',
                                     description='Script which creates graphs from AI recommender')
    parser.add_argument('--input_dir', dest='input_dir', type=str,
                        help='Folder where all input CSV files are located (Required)', required=True)
    parser.add_argument('--output_dir', dest='output_dir', type=str,
                        help='Folder where output graphs and updated CSV files will be saved (Required)', required=True)

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
    create_graphs(input_folder, output_folder)


if __name__ == "__main__":
    main()
