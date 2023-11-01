"""
Given a list of paths for data +  saved model folders, this file can be used to combine all of them together into a single excel file
"""

import argparse
import os

import pandas as pd

# calculation_types = [None, "inference_opposite_Evaluated", "infer_tile_images"]
calculation_types = ["inference_opposite_Evaluated_cumulative", "infer_tile_images_cumulative", "training"]
# calculation_types = ["training"]

lrlookup = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]  # TODO: needs a more flexible approach


def getallcsvs(path):
    allcsvs = [os.path.join(path, csvfile) for csvfile in os.listdir(path) if os.path.splitext(csvfile)[-1] == ".csv"]
    return allcsvs


def addallcolumns(columnlist, expt_folders, ms):
    newlist = []
    for folder in expt_folders:
        channel = os.path.basename(folder)
        subdirs = [os.path.join(folder, subdir) for subdir in os.listdir(folder) if
                   (os.path.isdir(os.path.join(folder, subdir)) and subdir.__contains__(ms))]
        for subdir in subdirs:
            if calculation_type is "training":
                calcsubdir = subdir
            else:
                calcsubdir = [os.path.join(subdir, calcsubdir) for calcsubdir in os.listdir(subdir) if
                              calcsubdir.__contains__(calculation_type)]
                # print(calcsubdir)
                assert len(
                    calcsubdir) <= 1, f"multiple folders for same calculation type :'{calculation_type}' found :{calcsubdir}"
                print("CC", subdir, expt_folders)
                [calcsubdir] = calcsubdir
            csvs = getallcsvs(calcsubdir)
            # print("csvs", calcsubdir)
            for csvfile in csvs:
                fname = os.path.basename(csvfile)
                try:  # metrics is after instead of before here
                    modelname, _, sampletype, instrument, lr, pretrained, _metricscsv = fname.split("_")
                except:
                    modelname, _, sampletype, instrument, _, lr, pretrained, _metricscsv = fname.split("_")
                # print(f"modelname, sampletype, instrument, lr, pretrained:"
                #       f"{modelname, sampletype, instrument, lr, pretrained}")

                data = pd.read_csv(csvfile, index_col=False)
                # print(f"datacols: {data.columns}")
                newlist = newlist + [dc for dc in list(data.columns) if dc.__contains__('Dice_')]
    # print("Clist :", columnlist)
    uniquecols = columnlist + sorted(list(set(newlist)))
    return uniquecols


def compile_folder(experiment_folders, model_substring, calculation_type="inference"):
    origcolumns = ['channel', 'modelname', 'sampletype', 'instrument', 'lr', 'pretrained', 'precision', 'recall',
                   'accuracy', 'F1-Score', 'Dice', 'Jaccard', 'MSE']
    columns = addallcolumns(origcolumns.copy(), experiment_folders, model_substring)
    combined = pd.DataFrame(columns=columns)
    # print(columns)
    c_row = 0
    for folder in experiment_folders:
        channel = os.path.basename(folder)
        # print("folder", folder)
        subdirs = [os.path.join(folder, subdir) for subdir in os.listdir(folder) if
                   (os.path.isdir(os.path.join(folder, subdir)) and subdir == model_substring)]
        for subdir in subdirs:
            if calculation_type is "training":
                calcsubdir = subdir
            else:
                calcsubdir = [os.path.join(subdir, calcsubdir) for calcsubdir in os.listdir(subdir) if
                              calcsubdir.__contains__(calculation_type)]
                assert len(calcsubdir) == 1, f"multiple folders for same calculation type :'{calculation_type}' found"
                [calcsubdir] = calcsubdir
            csvs = getallcsvs(calcsubdir)
            print("csvs", calcsubdir)
            for c, csvfile in enumerate(csvs):
                # model_INFER_sample_instrument_learning rate code_pretrained_metrics
                # deeplab50_INFER_PS_CG1D_1_ptFalse_metrics
                fname = os.path.basename(csvfile)
                # print(fname.split("_"))
                # TODO: based on location of "metrics'
                # try:  # metrics is after instead of before here
                modelname, _, sampletype, instrument, lr, pretrained, _metricscsv = fname.split("_")
                if _metricscsv != "metrics.csv":
                    modelname, _, _, sampletype, instrument, lr, pretrained = fname.split("_")

                data = pd.read_csv(csvfile, index_col=False)
                rows = len(data.index)
                for row in range(rows):
                    # print(calculation_type)
                    if calculation_type is "training":
                        accuracy = data['Per-Pixel Accuracy'][row]
                    else:
                        accuracy = data['Accuracy'][row]

                    combined.loc[c_row, origcolumns] = \
                        channel, modelname, sampletype, instrument, lrlookup[int(lr) - 1], pretrained.__contains__(
                            'True'), data['Precision'][row], accuracy, data['Recall'][row], \
                            (2 * data['Precision'][row] * data['Recall'][row]) / (
                                    data['Precision'][row] + data['Recall'][
                                row]), data['Dice'][row], data['Jaccard'][row], data['MSE'][row]

                    for ind_dice in sorted(list(set(columns) - set(origcolumns))):
                        combined.loc[c_row, str(ind_dice)] = data[ind_dice][row]
                    c_row += 1

    return combined


def compile_folder_train(experiment_folders, model_substring):
    columns = ['channel', 'modelname', 'sampletype', 'instrument', 'lr', 'pretrained', 'epoch', 'Train_loss',
               'Test_loss', 'precision', 'recall', 'accuracy', 'F1-Score', 'Dice', 'Jaccard', 'MSE']
    # columns = addallcolumns(columns, experiment_folders, model_substring)
    combined = pd.DataFrame(columns=columns)

    for folder in experiment_folders:
        channel = os.path.basename(folder)
        print("folder", folder)
        subdirs = [os.path.join(folder, subdir) for subdir in os.listdir(folder) if
                   (os.path.isdir(os.path.join(folder, subdir)) and subdir.__contains__(model_substring))]
        for subdir in subdirs:
            csvs = getallcsvs(subdir)
            for csvfile in csvs:
                fname = os.path.basename(csvfile)
                try:
                    modelname, _, _, sampletype, instrument, lr, pretrained = fname.split("_")
                except:
                    modelname, _, _, sampletype, instrument, _, lr, pretrained = fname.split("_")
                # print(f"modelname, sampletype, instrument, lr, pretrained:"
                #       f"{modelname, sampletype, instrument, lr, pretrained}")
                data = pd.read_csv(csvfile, index_col=False)
                rows = len(data.index)
                # print(data.columns)
                # print(combined.columns)
                for row in range(rows):
                    combined.loc[len(combined), combined.columns] = \
                        channel, modelname, sampletype, instrument, lrlookup[int(lr) - 1], pretrained.__contains__(
                            'True'), \
                            data['epoch'][row], data['Train_loss'][row], data['Test_loss'][row], data['Precision'][row], \
                            data['Per-Pixel Accuracy'][row], data['Recall'][row], data['F1-Score'][row], \
                            data['Dice'][row], data['Jaccard'][row], data['MSE'][row]

    return combined


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(prog='split',
    #                                  description='Script that renames image files according to mask files')
    # parser.add_argument('--foldernames', type=str, nargs='+', help='full path of main folder folder')
    # parser.add_argument('--model_folder_substring', type=str,
    #                     help='substring uniquely contained in names of all model folders')
    #
    # args, unknown = parser.parse_known_args()
    #
    # if args.foldername is None:
    #     print('ERROR: missing input mask folder ')
    # foldernames = args.foldernames
    # model_cstring = args.substring
    # folderpath = "C:/Users/pss2/NetBeansProjects/stats-simulations/data/ASD_3_4/MeasuredTrain"
    folderpath = "C:/Users/pss2/NetBeansProjects/stats-simulations/data/ASD34/MeasuredTrain"
    # folderpath = "C:/Users/pss2/NetBeansProjects/stats-simulations/data/CG1D_PS_comparechannels"
    foldernames = [
        f"{folderpath}/H0",
        f"{folderpath}/H0H1",
        f"{folderpath}/H0Hdark",
        f"{folderpath}/Hdark",
    ]
    model_cstring = "pytorchOutputMtoM_INFER_final"
    for calculation_type in calculation_types:
        # for calculation_type in [None]:
        print(calculation_type)
        df = None
        if calculation_type is None:
            calculation_type = "training"
            df = compile_folder_train(foldernames, model_cstring)
        else:
            df = compile_folder(foldernames, model_cstring, calculation_type=calculation_type)
        if df is not None:
            df.to_excel(os.path.join(folderpath, f"{calculation_type}_dic.xlsx"))
            # df.to_csv(os.path.join(folderpath, f"{calculation_type}_dic.csv"))
