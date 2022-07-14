# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import argparse
import os

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
'''
THis class is plotting parallel coordinate graphs of all attributes per AI model

The output is a merged file consisting of attributes about GPU utilization and AI model performance
as well as three parallel coordinate png images
 
'''
def merge_files(input_file1, input_file2, output_mergedfile):

    file1 = pd.read_csv(input_file1, sep=",")
    #print(file1)
    if file1.shape[0] < 20:
        print('ERROR: file1=', input_file1)
        print('ERROR - insufficient number of epochs < 20: df:', file1.shape[0])
        print('DEBUG: df=', file1)
        return False

    file2 = pd.read_csv(input_file2, sep=",")
    if file2.shape[0] < 20:
        print('ERROR: file2=', input_file1)
        print('ERROR - insufficient number of epochs < 20: df:', file2.shape[0])
        print('DEBUG: df=', file2)
        return False

    file3 = pd.merge(file1, file2, how='inner', left_on = 'CSV File Name', right_on = 'CSV File Name')
    print('merged:', file3)

    if file3.shape[0] < 20:
        print('ERROR: could not merge files: file1=', input_file1, ' file2=', input_file2)
        print('ERROR - insufficient number of epochs < 20: df:', file3.shape[0])
        print('DEBUG: df=', file3)
        return False

    basename_label = 'CSV File Name'
    basename_list = np.array(file3[basename_label][0:])
    # print('basename_list:', basename_list)
    plot_name = []
    for idx in range(0, len(basename_list)):
        parts = basename_list[idx].split("_")
        temp = ''
        for i in range(0,len(parts)):
            # ignore metrics in the filename
            if parts[i] == 'metrics':
                continue
            if parts[i] == 'fcn':
                continue
            # remove the file suffix
            if parts[i].endswith('.csv'):
                parts[i] = parts[i][:-4]
                # shorten the parameter
                if parts[i] == 'pretrained':
                    parts[i] = 'ini'
                temp += parts[i]
            else:
                temp += parts[i] + "_"

        plot_name.append(temp)

    file3.insert(0, "Plot name", plot_name)
    # print('file3:', file3)

    file3.to_csv(output_mergedfile, index=True, index_label='Model ID')
    return True

'''
This method selects the optimal configuration among all tested configurations per ai model architecture

'''
def find_opt_config(df,sort_label, select_label, min_flag):

    # sort by the sort label (ai model name)
    df_sort = df.sort_values(by=sort_label, ascending=True, na_position='first')
    # extract sort label (ai model architecture) and the select label (ce loss)
    plot_name_list = np.array(df_sort[sort_label][0:])
    accuracy_mintestloss_list = np.array(df_sort[select_label][0:])
    selected = []
    for i in range(0, len(plot_name_list)):
        architecture = plot_name_list[i].split('_')
        if i == 0:
            best_arch = architecture[0]
            best_error = accuracy_mintestloss_list[i]
            best_arch_sel = plot_name_list[i]
            continue
        if architecture[0] == best_arch:
            if min_flag and accuracy_mintestloss_list[i] < best_error:
                best_error = accuracy_mintestloss_list[i]
                best_arch = architecture[0]
                best_arch_sel = plot_name_list[i]
            if not min_flag and accuracy_mintestloss_list[i] > best_error:
                best_error = accuracy_mintestloss_list[i]
                best_arch = architecture[0]
                best_arch_sel = plot_name_list[i]
        else:
            # add to selected
            selected.append(best_arch_sel)
            # set the new best
            best_arch = architecture[0]
            best_error = accuracy_mintestloss_list[i]
            best_arch_sel = plot_name_list[i]

    # include the last architecture if it has not made it
    isIncluded = False
    for i in range(0, len(plot_name_list)):
        architecture = plot_name_list[i].split('_')
        if architecture[0] == best_arch_sel:
            isIncluded = True
    if not isIncluded:
        # add to selected
        selected.append(best_arch_sel)


    print('selected best architectures')
    for j in range(0, len(selected)):
        print('selected ', j, ' arch:', selected[j])

    return selected

'''
This method creates a paralell coordinate graph for 5 variables
'''
def create_parallel_coordinate_plot(df, model_id_label, plot_name_label, accuracy_mintestloss_label, data_uniformity_label, data_compatibility_model_label, training_predictPowerF_P20_label, output_file):
    model_id_list = np.array(df[model_id_label][0:])
    plot_name_list = np.array(df[plot_name_label][0:])
    #basename_list = np.array(df[basename_label][0:])

    accuracy_mintestloss_list = np.array(df[accuracy_mintestloss_label][0:])
    # accuracy_maxstability_list = np.array(df[accuracy_maxstability_label][0:])
    #
    # training_timemintestloss_list = np.array(df[training_timemintestloss_label][0:])
    # training_time100epochs_list = np.array(df[training_time100epochs_label][0:])
    ##########################

    # training_predictPowerF_P10_list = np.array(df[training_predictPowerF_P10_label][0:])
    training_predictPowerF_P20_list = np.array(df[training_predictPowerF_P20_label][0:])
    # training_predictExpF_P10_list = np.array(df[training_predictExpF_P10_label][0:])
    # training_predictExpF_P20_list = np.array(df[training_predictExpF_P20_label][0:])
    #
    # ################################
    data_uniformity_list = np.array(df[data_uniformity_label][0:])
    data_compatibility_model_list = np.array(df[data_compatibility_model_label][0:])
    #
    # avg_util_list = np.array(df[avg_util_label][0:])
    # avg_mem_util_list = np.array(df[avg_mem_util_label][0:])

    fig = go.Figure(data=
    go.Parcoords(
        line=dict(color=df[model_id_label],
                  colorscale=[[0, 'blue'], [0.5, 'green'], [1, 'red']], showscale=True),
        dimensions=list([
            dict(range=[np.min(accuracy_mintestloss_list), np.max(accuracy_mintestloss_list)],
                 # constraintrange=[1, 2],  # change this range by dragging the pink line
                 label='min:M_er', values=df[accuracy_mintestloss_label]),
            dict(range=[np.min(data_uniformity_list), np.max(data_uniformity_list)],
                 # tickvals=[1.5, 3, 4.5],
                 label='max:D_unif', values=df[data_uniformity_label]),
            dict(range=[np.min(model_id_list), np.max(model_id_list)],
                 tickvals=model_id_list,
                 label='M_id', values=df[model_id_label],
                 ticktext=plot_name_list),
            dict(range=[np.min(data_compatibility_model_list), np.max(data_compatibility_model_list)],
                 label='min:D_cm', values=df[data_compatibility_model_label]),
            dict(range=[np.min(training_predictPowerF_P20_list), np.max(training_predictPowerF_P20_list)],
                 label='min:P(PW_20)', values=df[training_predictPowerF_P20_label])
            # dict(range=[np.min(training_initgain_testloss_list), np.max(training_initgain_testloss_list)],
            #      label='I(Er)', values=df[training_initgain_testloss_label])
        ])
    )
    )

    fig.write_image(output_file)


def create_graphs(input_filename, output_folder):
    basename = os.path.basename(input_filename)
    print('INFO:: basename:', basename)

    model_id_label = 'Model ID'
    plot_name_label = 'Plot name'
    basename_label = 'CSV File Name'
    accuracy_mintestloss_label = 'Model-accuracy: Min Test Loss (min: best)'
    accuracy_maxstability_label = 'Model-stability: Sum NBH Test Loss (min: best)'

    training_timemintestloss_label = 'Training-speed: Time for Min Test Loss [s] (min: best)'
    training_time100epochs_label = 'Training-speed: Time for 100 Epochs [s] (min: best)'
    training_predictPowerF_P10_label = 'Training-predictability: Delta_PowerF_P10_Test (min: best)'
    training_predictPowerF_P20_label = 'Training-predictability: Delta_PowerF_P20_Test (min: best)'
    training_predictExpF_P10_label = 'Training-predictability: Delta_ExpF_P10_Test (min: best)'
    training_predictExpF_P20_label = 'Training-predictability: Delta_ExpF_P20_Test (min: best)'

    data_uniformity_label = 'Data-uniformity: Correlation(TrainLoss,TestLoss) (max abs value: best)'
    data_compatibility_model_label = 'Data-compatibility-model: Sum of integral under TrainLoss and TestLoss (min value: best)'

    training_initgain_testloss_label = 'Training-initgain: Delta of Min Test Loss without and with COCO initialization (pretrain) '
    training_initgain_time_label = 'Training-initgain: Delta of times for Min Test Loss without and with COCO initialization (pretrain) '

    data_compatibility_pretrain_label = 'Data-compatibility-pretrain: Sum of deltas equal to CE loss wihout and with pretrain for deltas > 0 (max value: best)'

    avg_util_label = 'average of GPU util. [%]'
    avg_mem_util_label = 'average of GPU Memory util. [%]'

    ###########################################################

    df_orig = pd.read_csv(input_filename)
    # import fields
    # CSV File Name	Model-accuracy: Min Test Loss (min: best)	Model-stability: Sum NBH Test Loss (min: best)	Training-speed: Time for Min Test Loss [s] (min: best)	Training-speed: Time for 100 Epochs [s] (min: best)	Training-predictability: Delta_PowerF_P10_Test (min: best)	Training-predictability: Delta_PowerF_P20_Test (min: best)	Training-predictability: Delta_ExpF_P10_Test (min: best)	Training-predictability: Delta_ExpF_P20_Test (min: best)	Data-uniformity: Correlation(TrainLoss,TestLoss) (max abs value: best)	Data-compatibility-model: Sum of integral under TrainLoss and TestLoss (min value: best)	Data-compatibility-pretrain: Sum of deltas equal to CE loss wihout and with pretrain for deltas > 0 (max value: best)	Training-initgain: Delta of Min Test Loss without and with COCO initialization (pretrain) 	Training-initgain: Delta of times for Min Test Loss without and with COCO initialization (pretrain)

    if df_orig.shape[0] < 20:
        print('ERROR: reading file:', input_filename)
        print('ERROR - insufficient number of epochs < 20: df:', df_orig.shape[0])
        print('DEBUG: df=', df_orig)
        return

    #######################################################################
    #df.sort_values(by=plot_name_label, ascending=True, na_position='first')
    # find optimal configuration based on CE loss
    min_flag = True # min is the best
    model_index_label = 'index'
    optimal_celoss_config = find_opt_config(df_orig, plot_name_label, accuracy_mintestloss_label, min_flag)
    df_select = df_orig.loc[df_orig[plot_name_label].isin(optimal_celoss_config)]
    # sort by the sort label
    df = df_select.sort_values(by=plot_name_label, ascending=True, na_position='first')
    # insert new indices
    a_list = list(range(0, len(df.index)))
    df.insert(0, model_index_label, a_list)
    print('selected table:', df)
    output_selectedfile = output_folder + os.path.sep + 'optimal_CEloss_configurations.csv'
    df.to_csv(output_selectedfile, index=False)

    basename = 'optimal_CEloss_configurations.png'
    outGraph_test = output_folder + os.path.sep + basename
    print('INFO: outGraph_test:', outGraph_test)
    create_parallel_coordinate_plot(df, model_index_label, plot_name_label, accuracy_mintestloss_label,
                                    data_uniformity_label, data_compatibility_model_label,
                                    training_predictPowerF_P20_label, outGraph_test)
    ############################################################
    # find optimal configuration based on data uniformity (correlation of train and test curves)
    min_flag = False  # max is the best
    optimal_config = find_opt_config(df_orig, plot_name_label, data_uniformity_label, min_flag)
    df_select = df_orig.loc[df_orig[plot_name_label].isin(optimal_config)]
    # sort by the sort label
    df = df_select.sort_values(by=plot_name_label, ascending=True, na_position='first')
    # insert new indices
    a_list = list(range(0, len(df.index)))
    df.insert(0, model_index_label, a_list)
    print('selected table:', df)
    output_selectedfile = output_folder + os.path.sep + 'optimal_Dunif_configurations.csv'
    df.to_csv(output_selectedfile, index=False)

    basename = 'optimal_Dunif_configurations.png'
    outGraph_test = output_folder + os.path.sep + basename
    print('INFO: outGraph_test:', outGraph_test)
    create_parallel_coordinate_plot(df, model_index_label, plot_name_label, accuracy_mintestloss_label,
                                    data_uniformity_label, data_compatibility_model_label,
                                    training_predictPowerF_P20_label, outGraph_test)

    ##########################################################################
    # find optimal configuration based on data-model compatibility (sum under train and test curves)
    min_flag = True  # min is the best
    optimal_config = find_opt_config(df_orig, plot_name_label, data_compatibility_model_label,min_flag)
    df_select = df_orig.loc[df_orig[plot_name_label].isin(optimal_config)]
    # sort by the sort label
    df = df_select.sort_values(by=plot_name_label, ascending=True, na_position='first')
    # insert new indices
    a_list = list(range(0, len(df.index)))
    df.insert(0, model_index_label, a_list)
    print('selected table:', df)
    output_selectedfile = output_folder + os.path.sep + 'optimal_DMcompat_configurations.csv'
    df.to_csv(output_selectedfile, index=False)

    basename = 'optimal_DMcompat_configurations.png'
    outGraph_test = output_folder + os.path.sep + basename
    print('INFO: outGraph_test:', outGraph_test)
    create_parallel_coordinate_plot(df, model_index_label, plot_name_label, accuracy_mintestloss_label,
                                    data_uniformity_label, data_compatibility_model_label,
                                    training_predictPowerF_P20_label, outGraph_test)

    ##########################################################################
    # find optimal configuration based on predictability P(PW-20) (delta between test CE loss and prediction
    min_flag = True  # min is the best
    optimal_config = find_opt_config(df_orig, plot_name_label, training_predictPowerF_P20_label,min_flag)
    df_select = df_orig.loc[df_orig[plot_name_label].isin(optimal_config)]
    # sort by the sort label
    df = df_select.sort_values(by=plot_name_label, ascending=True, na_position='first')
    # insert new indices
    a_list = list(range(0, len(df.index)))
    df.insert(0, model_index_label, a_list)
    print('selected table:', df)
    output_selectedfile = output_folder + os.path.sep + 'optimal_PW20_configurations.csv'
    df.to_csv(output_selectedfile, index=False)

    basename = 'optimal_PW20_configurations.png'
    outGraph_test = output_folder + os.path.sep + basename
    print('INFO: outGraph_test:', outGraph_test)
    create_parallel_coordinate_plot(df, model_index_label, plot_name_label, accuracy_mintestloss_label,
                                    data_uniformity_label, data_compatibility_model_label,
                                    training_predictPowerF_P20_label, outGraph_test)

    #########################################################################
    model_id_list = np.array(df[model_id_label][0:])
    plot_name_list = np.array(df[plot_name_label][0:])
    basename_list = np.array(df[basename_label][0:])

    accuracy_mintestloss_list = np.array(df[accuracy_mintestloss_label][0:])
    accuracy_maxstability_list = np.array(df[accuracy_maxstability_label][0:])

    training_timemintestloss_list = np.array(df[training_timemintestloss_label][0:])
    training_time100epochs_list = np.array(df[training_time100epochs_label][0:])
    ##########################

    training_predictPowerF_P10_list = np.array(df[training_predictPowerF_P10_label][0:])
    training_predictPowerF_P20_list = np.array(df[training_predictPowerF_P20_label][0:])
    training_predictExpF_P10_list = np.array(df[training_predictExpF_P10_label][0:])
    training_predictExpF_P20_list = np.array(df[training_predictExpF_P20_label][0:])

    ################################
    data_uniformity_list = np.array(df[data_uniformity_label][0:])
    data_compatibility_model_list = np.array(df[data_compatibility_model_label][0:])

    avg_util_list = np.array(df[avg_util_label][0:])
    avg_mem_util_list = np.array(df[avg_mem_util_label][0:])
    ###############################
    # TODO deal with a mixture of numerical and None values
    # training_initgain_testloss_list = np.array(df[training_initgain_testloss_label][0:])



    # inspired by https://plotly.com/python/parallel-coordinates-plot/
    #df = px.data.iris()
    # fig = px.parallel_coordinates(df, color=model_id_label, labels={accuracy_mintestloss_label: "M-er",
    #                                                               accuracy_maxstability_label: "M-st",
    #                                                               training_timemintestloss_label: "T(M-er)",
    #                                                               training_time100epochs_label: "T(100)",
    #                                                               training_predictPowerF_P10_label: "P(PW_P10)",
    #                                                               training_predictPowerF_P20_label: "P(PW_P20)",
    #                                                               training_predictExpF_P10_label: "P(EX_P10)",
    #                                                               training_predictExpF_P20_label: "P(EX_P20)",
    #                                                               training_initgain_testloss_label: "I(Er)",
    #                                                               training_initgain_time_label: "T(I(Er))",
    #                                                               data_uniformity_label: "D-Un",
    #                                                               data_compatibility_model_label: "D-C-M",
    #                                                               data_compatibility_pretrain_label: "D-C-P", },
    #                               color_continuous_scale=px.colors.diverging.Tealrose,
    #                               color_continuous_midpoint=2)


    fig = go.Figure(data=
    go.Parcoords(
        line=dict(color=df[model_id_label],
                  colorscale=[[0, 'blue'], [0.5, 'green'], [1, 'red']], showscale=True),
        dimensions=list([
            dict(range=[np.min(accuracy_mintestloss_list), np.max(accuracy_mintestloss_list)],
                 #constraintrange=[1, 2],  # change this range by dragging the pink line
                 label='M_er', values=df[accuracy_mintestloss_label]),
            dict(range=[np.min(accuracy_maxstability_list), np.max(accuracy_maxstability_list)],
                 #tickvals=[1.5, 3, 4.5],
                 label='M_stab', values=df[accuracy_maxstability_label]),
            dict(range=[np.min(model_id_list), np.max(model_id_list)],
                 tickvals=model_id_list,
                 label='M-id', values=df[model_id_label],
                 ticktext=plot_name_list),
            dict(range=[np.min(training_timemintestloss_list), np.max(training_timemintestloss_list)],
                 label='T(M_er)', values=df[training_timemintestloss_label]),
            dict(range=[np.min(training_time100epochs_list), np.max(training_time100epochs_list)],
                 label='T(100)', values=df[training_time100epochs_label])
            # dict(range=[np.min(training_initgain_testloss_list), np.max(training_initgain_testloss_list)],
            #      label='I(Er)', values=df[training_initgain_testloss_label])
        ])
    )
    )

    basename = 'optimal_model.png'
    outGraph_test = output_folder + os.path.sep + basename
    print('INFO: outGraph_test:', outGraph_test)
    fig.write_image(outGraph_test)

    # fig.show()
    #########################################################################
    fig2 = go.Figure(data=
    go.Parcoords(
        line=dict(color=df[model_id_label],
                  colorscale=[[0, 'blue'], [0.5, 'green'], [1, 'red']], showscale=True),
        dimensions=list([
            dict(range=[np.min(training_predictPowerF_P10_list), np.max(training_predictPowerF_P10_list)],
                 #constraintrange=[1, 2],  # change this range by dragging the pink line
                 label='P(PW_10)', values=df[training_predictPowerF_P10_label]),
            dict(range=[np.min(training_predictPowerF_P20_list), np.max(training_predictPowerF_P20_list)],
                 #tickvals=[1.5, 3, 4.5],
                 label='P(PW_20)', values=df[training_predictPowerF_P20_label]),
            dict(range=[np.min(model_id_list), np.max(model_id_list)],
                 tickvals=model_id_list,
                 label='M_id', values=df[model_id_label],
                 ticktext=plot_name_list),
            dict(range=[np.min(training_predictExpF_P10_list), np.max(training_predictExpF_P10_list)],
                 label='P(EX_10)', values=df[training_predictExpF_P10_label]),
            dict(range=[np.min(training_predictExpF_P20_list), np.max(training_predictExpF_P20_list)],
                 label='P(EX_20)', values=df[training_predictExpF_P20_label])
        ])
    )
    )

    basename = 'optimal_process.png'
    outGraph_test = output_folder + os.path.sep + basename
    print('INFO: outGraph_test:', outGraph_test)
    fig2.write_image(outGraph_test)
    #########################################################################
    fig3 = go.Figure(data=
    go.Parcoords(
        # line=dict(color=df[model_id_label],
        #           colorscale='Electric',
        #           showscale=True,
        #           cmin=0,
        #           cmax=11),
        line=dict(color=df[model_id_label],
                  colorscale=[[0, 'blue'], [0.5, 'green'], [1, 'red']], showscale=True),
        dimensions=list([
            dict(range=[np.min(data_uniformity_list), np.max(data_uniformity_list)],
                 #constraintrange=[1, 2],  # change this range by dragging the pink line
                 label='D_unif', values=df[data_uniformity_label]),
            dict(range=[np.min(data_compatibility_model_list), np.max(data_compatibility_model_list)],
                 #tickvals=[1.5, 3, 4.5],
                 label='D_cm', values=df[data_compatibility_model_label]),
            dict(range=[np.min(model_id_list), np.max(model_id_list)],
                 tickvals=model_id_list,
                 label='M_id', values=df[model_id_label],
                 ticktext=plot_name_list),
            dict(range=[np.min(avg_util_list), np.max(avg_util_list)],
                 label='GPU_u', values=df[avg_util_label]),
            dict(range=[np.min(avg_mem_util_list), np.max(avg_mem_util_list)],
                 label='GPU_m', values=df[avg_mem_util_label])
        ])
    )
    )

    basename = 'optimal_data.png'
    outGraph_test = output_folder + os.path.sep + basename
    print('INFO: outGraph_test:', outGraph_test)
    fig3.write_image(outGraph_test)

def main():
    # Setup the Argument parsing
    parser = argparse.ArgumentParser(prog='create_graphs', description='Script which creates graphs from AI recommender')
    parser.add_argument('--input_dir', dest='input_dir', type=str, help='Folder where all input CSV files are located (Required)', required=True)
    parser.add_argument('--output_dir', dest='output_dir', type=str, help='Folder where output graphs and updated CSV files will be saved (Required)', required=True)

    args = parser.parse_args()

    if args.input_dir is None:
        print('ERROR: missing input  dir ')
        return

    if args.output_dir is None:
        print('ERROR: missing output dir ')
        return

    input_folder = args.input_dir
    output_folder = args.output_dir

    # print('Arguments:')
    # print('input folder = {}'.format(input_folder))
    # print('output folder = {}'.format(output_folder))

    input_file1 = input_folder + os.path.sep + 'metrics.csv'
    #'/home/pnb/trainingOutput/pytorchOutput_A10/comparisons/metrics.csv'
    input_file2 = input_folder + os.path.sep + 'gpu_stats.csv'
    #'/home/pnb/trainingOutput/pytorchOutput_A10/comparisons/gpu_stats.csv'
    output_mergedfile = output_folder + os.path.sep + 'merged_metrics.csv'
    #'/home/pnb/trainingOutput/pytorchOutput_A10/comparisons/merged_metrics.csv'
    ret = merge_files(input_file1, input_file2, output_mergedfile)
    if ret:
        create_graphs(output_mergedfile, output_folder)


if __name__ == "__main__":
    main()
