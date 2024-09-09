"""
Given a list of paths for data +  saved model folders, this file can be used to combine all of them together into a single excel file


"""

import argparse
import os
import pandas as pd



# lrlookup = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]  # TODO: needs a more flexible approach
lrlookup = [1e-3, 1e-2]


def getallcsvs(path, ignore_old=True):
    allcsvs = [os.path.join(path, csfile) for csfile in os.listdir(path) if os.path.splitext(csfile)[-1] == ".csv"]
    if ignore_old:
        allcsvs = [cfile for cfile in allcsvs if not "old" in cfile]
    return allcsvs


def addallcolumns(columnlist, expt_folders, ms, calculation_type):
    newlist = []
    for folder in expt_folders:
        channel = os.path.basename(folder)
        subdirs = [os.path.join(folder, subdir) for subdir in os.listdir(folder) if
                   (os.path.isdir(os.path.join(folder, subdir)) and subdir.__contains__(ms))]
        for subdir in subdirs:
            if calculation_type == "training":
                calcsubdir = subdir
            else:
                calcsubdir = [os.path.join(subdir, c) for c in os.listdir(subdir) if
                              c.__contains__(calculation_type)]
                # print(calcsubdir)
                if len(calcsubdir) > 1:
                    print(subdir, calcsubdir)
                    calcsubdir = [os.path.join(subdir, c) for c in os.listdir(subdir) if
                                  c == calculation_type]
                    # , f"multiple folders for same calculation type :'{calculation_type}' found :{calcsubdir}"
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

                data = pd.read_csv(csvfile, index_col=False)
                # print(f"datacols: {data.columns}")
                newlist = newlist + [dc for dc in list(data.columns) if dc.__contains__('Dice_')]
    # print("Clist :", columnlist)
    uniquecols = columnlist + sorted(list(set(newlist)))
    return uniquecols


def compile_folder(experiment_folders, model_substring, calculation_type="inference"):
    origcolumns = ['channel', 'modelname', 'sampletype', 'instrument', 'lr', 'pretrained', 'precision', 'recall',
                   'accuracy', 'F1-Score', 'Dice', 'Jaccard', 'MSE']
    columns = addallcolumns(origcolumns.copy(), experiment_folders, model_substring, calculation_type)
    combined = pd.DataFrame(columns=columns)
    # print(columns)
    c_row = 0
    for folder in experiment_folders:
        channel = os.path.basename(folder)
        # print("folder", folder)
        subdirs = [os.path.join(folder, subdir) for subdir in os.listdir(folder) if
                   (os.path.isdir(os.path.join(folder, subdir)) and subdir == model_substring)]
        for subdir in subdirs:
            if calculation_type == "training":
                calcsubdir = subdir
            else:
                calcsubdir = [os.path.join(subdir, calcsubdir) for calcsubdir in os.listdir(subdir) if
                              calcsubdir.__contains__(calculation_type)]
                # assert len(calcsubdir) == 1, f"multiple folders for same calculation type :'{calculation_type}' found"
                if len(calcsubdir) > 1:
                    print(subdir, calcsubdir)
                    calcsubdir = [os.path.join(subdir, c) for c in os.listdir(subdir) if
                                  c == calculation_type]
                    # , f"multiple folders for same calculation type :'{calculation_type}' found :{calcsubdir}"

                [calcsubdir] = calcsubdir
            csvs = getallcsvs(calcsubdir)
            print("csvs", calcsubdir)
            for c, csvfile in enumerate(csvs):
                fname = os.path.basename(csvfile)
                modelname, _, sampletype, instrument, lr, pretrained, _metricscsv = fname.split("_")
                if _metricscsv != "metrics.csv":
                    modelname, _, _, sampletype, instrument, lr, pretrained = fname.split("_")

                data = pd.read_csv(csvfile, index_col=False)
                rows = len(data.index)
                for row in range(rows):
                    if calculation_type == "training":
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
                data = pd.read_csv(csvfile, index_col=False)
                rows = len(data.index)
                for row in range(rows):
                    combined.loc[len(combined), combined.columns] = \
                        channel, modelname, sampletype, instrument, lrlookup[int(lr) - 1], pretrained.__contains__(
                            'True'), \
                            data['epoch'][row], data['Train_loss'][row], data['Test_loss'][row], data['Precision'][row], \
                            data['Per-Pixel Accuracy'][row], data['Recall'][row], data['F1-Score'][row], \
                            data['Dice'][row], data['Jaccard'][row], data['MSE'][row]

    return combined


def main():
    parser = argparse.ArgumentParser(prog='split',
                                     description='Script that renames image files according to mask files')
    parser.add_argument('--folderpath', type=str, nargs='+', help='full path of main folder folder')
    parser.add_argument('--calculation_types', type=str, nargs='+', default='training',
                        help='csv name. training and inference results are treated differently, list all folder name substrings. e.g. inference_opposite_evaluated')
    parser.add_argument('--model_cstring', type=str, default="pytorchOutputMtoM_INFER_final",
                        help='substring uniquely contained in names of all model folders')

    args, unknown = parser.parse_known_args()

    if args.foldername is None:
        print('ERROR: missing input mask folder ')
    folderpath = args.folderpath
    # model_cstring = args.substring
    # folderpath = "E:/Data/INFER/PBS/LANL_PBSDDS_Clean_5_10/Combined/9-1_PBS-DDS"
    # folderpath = "E:/Data/INFER/PBS/LANL_PBSDDS_Clean_5_10/Combined/10-0_PBS-DDS"
    foldernames = [
        # f"{folderpath}/H0",
        f"{folderpath}",
        # f"{folderpath}/H0H1",
        # f"{folderpath}/H0Hdark",
        # f"{folderpath}/Hdark",
    ]
    model_cstring = args.model_cstring
    calculation_types = args.calculation_types
    for calculation_type in calculation_types:
        # for calculation_type in [None]:
        print("calculation type: ", calculation_type)
        df = None
        if calculation_type is None:
            calculation_type = "training"
            df = compile_folder_train(foldernames, model_cstring)
        else:
            df = compile_folder(foldernames, model_cstring, calculation_type=calculation_type)
        if df is not None:
            df.to_excel(os.path.join(folderpath, f"{calculation_type}_dic.xlsx"))
            # df.to_csv(os.path.join(folderpath, f"{calculation_type}_dic.csv"))


if __name__ == "__main__":
    main()
