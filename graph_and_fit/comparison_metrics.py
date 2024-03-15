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
this class is for comparing multiple AI models, training data, and training processes
using the data gathered by AI recommender
__author__      = "Peter Bajcsy"
__email__ = "peter.bajcsy@nist.gov"
'''

'''
This is the main routine for computing comparison metrics
'''


def compute_comparison_metrics(in_path, out_path):
    # Check whether the specified output path exists or not
    doesExist = os.path.exists(out_path)
    if not doesExist:
        os.makedirs(out_path)

    # create header for the output metrics.csv file
    output_metrics_file = out_path + os.path.sep + "metrics.csv"
    print('INFO: output_metrics_file:', output_metrics_file)

    basename_label = 'CSV File Name'
    accuracy_mintestloss_label = 'Model-accuracy: Min Test Loss (min: best)'
    accuracy_maxstability_label = 'Model-stability: Sum NBH Test Loss (min: best)'

    training_timemintestloss_label = 'Training-speed: Time for Min Test Loss [s] (min: best)'
    training_time100epochs_label = 'Training-speed: Time for 100 Epochs [s] (min: best)'
    training_predictPowerF_P10_label = 'Training-predictability: Delta_PowerF_P10_Test (min: best)'
    training_predictPowerF_P20_label = 'Training-predictability: Delta_PowerF_P20_Test (min: best)'
    training_predictExpF_P10_label = 'Training-predictability: Delta_ExpF_P10_Test (min: best)'
    training_predictExpF_P20_label = 'Training-predictability: Delta_ExpF_P20_Test (min: best)'

    training_initgain_testloss_label = 'Training-initgain: Delta of Min Test Loss without and with COCO initialization (pretrain) '
    training_initgain_time_label = 'Training-initgain: Delta of times for Min Test Loss without and with COCO initialization (pretrain) '

    data_uniformity_label = 'Data-uniformity: Correlation(TrainLoss,TestLoss) (max abs value: best)'
    data_compatibility_model_label = 'Data-compatibility-model: Sum of integral under TrainLoss and TestLoss (min value: best)'
    data_compatibility_pretrain_label = 'Data-compatibility-pretrain: Sum of deltas equal to CE loss wihout and with pretrain for deltas > 0 (max value: best)'

    metrics_list = pd.DataFrame(columns=[basename_label, accuracy_mintestloss_label, accuracy_maxstability_label,
                                         training_timemintestloss_label, training_time100epochs_label,
                                         training_predictPowerF_P10_label,
                                         training_predictPowerF_P20_label, training_predictExpF_P10_label,
                                         training_predictExpF_P20_label,
                                         data_uniformity_label, data_compatibility_model_label,
                                         data_compatibility_pretrain_label, training_initgain_testloss_label,
                                         training_initgain_time_label])
    metrics_list.to_csv(output_metrics_file, mode='w', header=True, index=False)

    csv_folder = os.path.abspath(in_path)
    for csv_file in os.listdir(csv_folder):
        if csv_file.endswith(".csv"):
            print(os.path.join(in_path, csv_file))

    basename_metrics = []
    accuracy_mintestloss = []
    accuracy_maxstability = []

    training_timemintestloss = []
    training_time100epochs = []
    training_predictable_powerF_P10 = []
    training_predictable_powerF_P20 = []
    training_predictable_expF_P10 = []
    training_predictable_expF_P20 = []

    training_initgain_testloss = []
    training_initgain_time = []

    data_uniformity = []
    data_compatibility_model = []
    data_compatibility_pretrain = []

    os.chdir(in_path)
    for fname in glob.glob("*.csv"):
        print('INFO: fname:', fname)
        basename = os.path.basename(fname)
        print('INFO:: comparison_metrics basename:', basename)

        if basename.endswith('coefficients.csv'):
            # skip the summary file named coefficients.csv
            continue

        ###########################################################
        basename_metrics.append(basename)

        df = pd.read_csv(fname)
        if df.shape[0] < 20:
            print('ERROR in comparison_metrics: reading file:', fname)
            print('ERROR - insufficient number of epochs < 20: dataF:', df.shape[0])
            print('DEBUG: dataF=', df)
            continue
        # gather data for power function fit from the CSV file
        x_final = np.array(df['Seconds'][0:])
        train_loss = np.array(df['Train_loss'][0:])
        test_loss = np.array(df['Test_loss'][0:])
        ##########################################
        # must filter points to avoid parallel plot showing one huge number of misfit (and not supporting log scale)
        # from TrojAI doc web site: https://pages.nist.gov/trojai/docs/overview.html
        # the same max value is applied in power_function_fit.py
        max_ce_value = 36
        train_loss = np.clip(train_loss, 0, max_ce_value)
        test_loss = np.clip(test_loss, 0, max_ce_value)
        ###############################

        delta_powerF_P10_test = np.array(df['Delta_PowerF_P10_Test'][0:])
        delta_powerF_P20_test = np.array(df['Delta_PowerF_P20_Test'][0:])
        delta_expF_P10_test = np.array(df['Delta_ExpF_P10_Test'][0:])
        delta_expF_P20_test = np.array(df['Delta_ExpF_P20_Test'][0:])

        ########################################
        # Metrics
        # 1. best model accuracy - find min
        min_train_loss = np.amin(train_loss)
        min_test_loss = np.amin(test_loss)
        print('INFO: min_test_loss:', min_test_loss)

        accuracy_mintestloss.append(min_test_loss)
        #####################
        # epoch index of the best model accuracy
        min_index_train_loss = np.where(train_loss == min_train_loss)
        min_index_test_loss = np.where(test_loss == min_test_loss)
        # if len(compute_comparison_metrics) > 1:
        #     print('INFO: len(min_index_test_loss) > 1:', len(min_index_test_loss))

        print('INFO: min_index_test_loss:', min_index_test_loss[0].data[0])
        # accuracy_mintestloss_label = 1 #'accuracy_mintestloss_label'
        # array_metrics[accuracy_mintestloss_label] = min_index_test_loss[0]
        # accuracy_minindextestloss.append(min_index_test_loss[0].data[0])
        ########################################
        # 2. model stability - find min
        stable_metric = 0
        stabledelta_nbh = 5
        if 2 * stabledelta_nbh > len(test_loss):
            stabledelta_nbh = len(test_loss) / 2

        min_range = min_index_test_loss[0] - stabledelta_nbh
        max_range = min_index_test_loss[0] + stabledelta_nbh
        if min_range < 0:
            min_range = 0
            max_range = min_index_test_loss[0] + 2 * stabledelta_nbh
        if max_range >= len(test_loss):
            max_range = len(test_loss)
            min_range = max_range - 2 * stabledelta_nbh

        print('INFO: min_range [epochs]:', min_range)
        print('INFO: max_range [epochs]:', max_range)
        for i in range(int(min_range), int(max_range)):
            stable_metric += np.abs(test_loss[i] - min_train_loss)

        # accuracy_stability_label = 2 # 'accuracy_stability' + int(stabledelta_nbh) + '_label'
        # array_metrics[accuracy_stability_label] = stable_metric
        accuracy_maxstability.append(stable_metric)
        ########################################
        # 3. time needed to train to reach the best model accuracy
        # traning process speed - find min
        time_min_test_loss = x_final[min_index_test_loss[0]]
        print('INFO: time_min_test_loss [seconds]:', time_min_test_loss.data[0])
        # training_time_mintestloss_label = 3 # 'training_time_mintestloss_label'
        # array_metrics[training_time_mintestloss_label] = time_min_test_loss
        training_timemintestloss.append(time_min_test_loss.data[0])

        time_train_100epochs = x_final[len(x_final) - 1]  # np.amax(x_final) # - find min
        print('INFO: time_train_100epochs [seconds]:', time_train_100epochs)
        # training_time_100epochs_label = 4 # 'training_time_100epochs_label'
        # array_metrics[training_time_100epochs_label] = time_train_100epochs
        training_time100epochs.append(time_train_100epochs)

        ########################################
        # 4. predictable of training process - find min
        predictable_error_powerF_P10 = np.sum(abs(delta_powerF_P10_test))
        print('INFO: predictable_error (powerF_P10_test):', predictable_error_powerF_P10)
        training_predictable_powerF_P10.append(predictable_error_powerF_P10)

        predictable_error_powerF_P20 = np.sum(abs(delta_powerF_P20_test))
        print('INFO: predictable_error (powerF_P20_test):', predictable_error_powerF_P20)
        training_predictable_powerF_P20.append(predictable_error_powerF_P20)

        predictable_error_expF_P10 = np.sum(abs(delta_expF_P10_test))
        print('INFO: predictable_error (expF_P10_test):', predictable_error_powerF_P10)
        training_predictable_expF_P10.append(predictable_error_expF_P10)

        predictable_error_expF_P20 = np.sum(abs(delta_expF_P20_test))
        print('INFO: predictable_error (expF_P20_test):', predictable_error_expF_P20)
        training_predictable_expF_P20.append(predictable_error_expF_P20)

        ########################################
        # Pearson product-moment correlation coefficients according to
        # https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html
        # 5. intra-data uniformity - find max
        intra_data_uniformity = np.corrcoef(train_loss, test_loss)
        print('INFO: intra_data_uniformity:', intra_data_uniformity[0][1])
        # data_uniformity_label = 6 #'data_uniformity_label'
        # array_metrics[data_uniformity_label] = intra_data_uniformity
        data_uniformity.append(intra_data_uniformity[0][1])

        ########################################
        # 6. inter-data compatibility = sum of integrals under trainCE and testCE curves
        sum_trainCE = np.sum(train_loss)
        sum_testCE = np.sum(test_loss)

        print('sum_trainCE:', sum_trainCE)
        print('sum_testCE:', sum_testCE)
        data_compatibility_model.append(sum_trainCE + sum_testCE)

        #######################################
        # 7. pair-wise metric for pretrained and from scratch models
        if basename.endswith('_pretrained.csv'):
            # compute the pair-wise comparison of COCO pretrained and from scratch
            parts = basename.split("_")
            # this should never happen because of the first if statement
            if len(parts) < 1:
                print('ERROR: CSV files do not contain _ delimiter separating AI model from config parameters')
                continue

            print('parts[0] ', parts[0], ' parts[1]:', parts[1], ' parts[-2]:', parts[-2])
            # Note: must match two elements because of naming convention fcn_resnet50 and fcn_resnet101
            start_match_string = parts[0] + "_" + parts[1]
            contain_match_string = "_" + parts[-2] + ".csv"
            foundMatch = False
            for csv_file in os.listdir(csv_folder):
                # print('DEBUG: csv_file:', csv_file)
                if csv_file.endswith(".csv") and csv_file.startswith(start_match_string) and \
                        not csv_file.endswith('_pretrained.csv') and contain_match_string in csv_file:
                    fname_match_pretrained = os.path.join(in_path, csv_file)
                    print('fname_match_pretrained:', fname_match_pretrained)
                    foundMatch = True

                    df_pretrained = pd.read_csv(fname_match_pretrained)
                    # gather data for power function fit from the CSV file
                    # x_final_pretrained = np.array(df_pretrained['Seconds'][0:])
                    # train_loss_pretrained = np.array(df_pretrained['Train_loss'][0:])
                    test_loss_pretrained = np.array(df_pretrained['Test_loss'][0:])

                    min_test_loss_pretrained = np.amin(test_loss_pretrained)
                    print('INFO: min_test_loss_pretrained:', min_test_loss_pretrained)
                    # epoch index of the best model accuracy
                    min_index_test_loss_pretrained = np.where(test_loss_pretrained == min_test_loss_pretrained)
                    print('INFO: min_index_test_loss_pretrained:', min_index_test_loss_pretrained)

                    training_initgain_testloss.append(min_test_loss - min_test_loss_pretrained)
                    training_initgain_time.append(
                        min_index_test_loss[0].data[0] - min_index_test_loss_pretrained[0].data[0])

                    sum_comp = 0
                    for i in range(0, len(test_loss)):
                        if test_loss[i] > test_loss_pretrained[i]:
                            sum_comp += test_loss[i] - test_loss_pretrained[i]

                    data_compatibility_pretrain.append(sum_comp)

            if not foundMatch:
                print('WARNING: the pretrained file does not have a matching non-pretrained file:', basename)
                # Training initialization gain
                training_initgain_testloss.append('None')
                training_initgain_time.append('None')
                data_compatibility_pretrain.append('None')

        else:
            # Training initialization gain
            training_initgain_testloss.append('None')
            training_initgain_time.append('None')
            data_compatibility_pretrain.append('None')

    metrics_list[basename_label] = basename_metrics
    metrics_list[accuracy_mintestloss_label] = accuracy_mintestloss
    metrics_list[accuracy_maxstability_label] = accuracy_maxstability

    metrics_list[training_timemintestloss_label] = training_timemintestloss
    metrics_list[training_time100epochs_label] = training_time100epochs

    metrics_list[training_predictPowerF_P10_label] = training_predictable_powerF_P10
    metrics_list[training_predictPowerF_P20_label] = training_predictable_powerF_P20
    metrics_list[training_predictExpF_P10_label] = training_predictable_expF_P10
    metrics_list[training_predictExpF_P20_label] = training_predictable_expF_P20

    metrics_list[data_uniformity_label] = data_uniformity
    metrics_list[data_compatibility_model_label] = data_compatibility_model

    metrics_list[data_compatibility_pretrain_label] = data_compatibility_pretrain
    metrics_list[training_initgain_testloss_label] = training_initgain_testloss
    metrics_list[training_initgain_time_label] = training_initgain_time

    ##########################################
    # metrics_list = pd.DataFrame({"metrics": [array_metrics]})
    print(metrics_list)
    # metrics_list.to_csv(output_metrics_file, index=False)
    metrics_list.to_csv(output_metrics_file, mode='a', header=False, index=False)


def main():
    # Setup the Argument parsing
    parser = argparse.ArgumentParser(prog='compare_metrics',
                                     description='Script which computes metrics for comparing AI architectures')
    parser.add_argument('--input_dir', dest='input_dir', type=str,
                        help='Folder where all input CSV files are located (Required)', required=True)
    parser.add_argument('--output_dir', dest='output_dir', type=str,
                        help='Folder where output metrics will be saved (Required)', required=True)

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
    compute_comparison_metrics(input_folder, output_folder)


if __name__ == "__main__":
    main()
