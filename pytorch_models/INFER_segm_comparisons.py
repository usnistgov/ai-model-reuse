# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import csv
import argparse
from pathlib import Path
import time
import skimage.io
import skimage
import numpy as np
from sklearn.metrics import f1_score, mean_squared_error, jaccard_score

# TODO: replace precision,recall,f1?
'''
compute the accuracy against the ground truth mask over a folder of predicted and ground truth masks
__author__      = "Peter Bajcsy"
__email__ = "peter.bajcsy@nist.gov"
'''


def compare_batch(pred_mask_dirpath, gt_mask_dirpath, gt_mask_numclasses, output_dir, doNamesMatch):
    start_time = time.time()
    file_array = []
    for filename in os.listdir(pred_mask_dirpath):
        filepath = os.path.join(pred_mask_dirpath, filename)
        file_array.append(filepath)

    mask_file_array = []
    for filename in os.listdir(gt_mask_dirpath):
        filepath = os.path.join(gt_mask_dirpath, filename)
        mask_file_array.append(filepath)

    if len(file_array) != len(mask_file_array):
        print('INFO: number of files in predicted and gt mask folders is different: pred=', len(file_array), ' gt=',
              len(mask_file_array))

    avg_precision_score = 0.0
    avg_recall_score = 0.0
    avg_accuracy_score = 0.0
    avg_f1_score = 0.0
    for i in range(len(file_array)):
        image_basename = os.path.basename(file_array[i])
        if doNamesMatch:
            # perform matching
            found_match = False
            match_index = 0
            for j in range(len(mask_file_array)):
                mask_basename = os.path.basename(mask_file_array[j])
                if mask_basename == image_basename:
                    match_index = j
                    found_match = True
                    break
            if not found_match:
                print('INFO: did not find a matching mask for ', file_array[i])
                continue
        else:
            if i < len(mask_file_array):
                match_index = i
            else:
                match_index = len(mask_file_array) - 1
                print('INFO: this predicted mask is compared against the last gt mask:', image_basename)
                # continue

        # load predicted mask
        pred_mask = skimage.io.imread(file_array[i])
        print('INFO: loaded pred mask - ', image_basename)
        gt_mask = skimage.io.imread(mask_file_array[match_index])
        mask_basename = os.path.basename(mask_file_array[match_index])
        print('INFO: loaded gt mask - ', mask_basename)

        if pred_mask.shape[0] != gt_mask.shape[0] or pred_mask.shape[1] != gt_mask.shape[1]:
            print('ERROR: sanity check - prediction and gt masks do not have the same dimensions')
            continue

        precision_score, recall_score, accuracy_score, f1_score = metrics_masks(mask_file_array[match_index], pred_mask,
                                                                                gt_mask, gt_mask_numclasses, output_dir)
        avg_precision_score += precision_score
        avg_recall_score += recall_score
        avg_accuracy_score += accuracy_score
        avg_f1_score += f1_score
        ##########################

    if len(file_array) > 0:
        avg_precision_score /= len(file_array)
        avg_recall_score /= len(file_array)
        avg_accuracy_score /= len(file_array)
        avg_f1_score /= len(file_array)

    fieldnames = ['GTMask_filepath', 'PredMask_filepath', 'Precision', 'Recall', 'Accuracy', 'F1-Score',
                  'Exec_time [seconds]']
    metrics_name = 'summary_accuracy.csv'
    path_to_file = os.path.join(output_dir, metrics_name)
    print('INFO: summary output stats:', path_to_file)
    file_exists = os.path.exists(path_to_file)
    if not file_exists:
        with open(path_to_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    batchsummary = {a: [0] for a in fieldnames}
    batchsummary['GTMask_filepath'] = gt_mask_dirpath
    batchsummary['PredMask_filepath'] = pred_mask_dirpath
    batchsummary['Precision'] = avg_precision_score
    batchsummary['Recall'] = avg_recall_score
    batchsummary['Accuracy'] = avg_accuracy_score
    batchsummary['F1-Score'] = avg_f1_score
    exec_time = time.time() - start_time
    exec_time = int(exec_time)
    batchsummary['Exec_time [seconds]'] = exec_time
    print(batchsummary)
    with open(os.path.join(output_dir, metrics_name), 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(batchsummary)


def metrics_masks(gt_file_name, pred_mask, gt_mask, num_classes, output_dir):
    start_time = time.time()
    fieldnames = ['Mask_name', 'Precision', 'Recall', 'Accuracy', 'F1-Score', 'Exec_time [seconds]']

    conf_matrix = confusion_matrix(pred_mask, gt_mask, num_classes)

    # Precision Score = TP / (FP + TP)
    row_sums = np.sum(conf_matrix, 0)
    running_precision = 0
    avgsubtract = 0
    for i in range(0, num_classes):
        toadd = 0
        if row_sums[i] == 0:
            toadd = 0
            avgsubtract += 1
        else:
            toadd = conf_matrix[i][i] / row_sums[i]
        running_precision += toadd

    precision_score = running_precision / (num_classes - avgsubtract)

    # Recall Score = TP / (FN + TP)
    col_sums = np.sum(conf_matrix, axis=1)
    running_recall = 0
    avgsubtract = 0
    for i in range(0, num_classes):
        toadd = 0
        if col_sums[i] == 0:
            toadd = 0
            avgsubtract += 1
        else:
            toadd = conf_matrix[i][i] / col_sums[i]
        running_recall += toadd
    recall_score = running_recall / (num_classes - avgsubtract)

    # F1 Score = 2* Precision Score * Recall Score/ (Precision Score + Recall Score/)
    # mean_f1_score = 2.0 * (mean_precision_score * mean_recall_score) / (mean_precision_score + mean_recall_score)
    f1_score = 2.0 * (precision_score * recall_score) / (precision_score + recall_score)

    # Accuracy Score = (TP + TN) / (TP + FN + TN + FP)
    tp_and_tn = 0.0
    total_count = 0.0
    for h in range(conf_matrix.shape[0]):
        for w in range(conf_matrix.shape[1]):
            if h == w:
                tp_and_tn = tp_and_tn + conf_matrix[h][w]
            total_count = total_count + conf_matrix[h][w]
    if total_count > 0:
        accuracy_score = tp_and_tn / total_count
    else:
        accuracy_score = 0.0
    seconds = time.time() - start_time
    seconds = int(seconds)
    # print()
    # print(f"1: {np.unique(gt_mask).tolist()}, {np.unique(pred_mask).tolist()}")
    # print(f"2: {np.unique(gt_mask).tolist() + np.unique(pred_mask).tolist()}")
    # print(f"3: {set(np.unique(gt_mask).tolist() + np.unique(pred_mask).tolist())}")
    # print(f"4: {list(set(np.unique(gt_mask).tolist() + np.unique(pred_mask).tolist()))}")
    all_labels = sorted(list(set(np.unique(gt_mask).tolist() + np.unique(pred_mask).tolist())))
    labelwise_dice = {}
    labelwise_jaccard = jaccard_score(gt_mask.flatten(), pred_mask.flatten(), labels=all_labels, average=None)
    for l, label in enumerate(all_labels):
        labelwise_dice[f'{label}'] = 2 * labelwise_jaccard[l] / (labelwise_jaccard[l] + 1)

    jaccard_coeff = jaccard_score(gt_mask.flatten(), pred_mask.flatten(), average='micro')
    dice_coeff = (2 * jaccard_coeff) / (jaccard_coeff + 1)
    mse_coeff = mean_squared_error(gt_mask.flatten(), pred_mask.flatten())
    return precision_score, recall_score, accuracy_score, f1_score, jaccard_coeff, dice_coeff, mse_coeff, labelwise_dice


def confusion_matrix(predictions, masks, classes):
    matrix = np.zeros((classes, classes))
    for h in range(predictions.shape[0]):
        for w in range(predictions.shape[1]):
            # if predictions[h][w] == 1 and masks[h][w] == 1:
            #     print('DEBUG')
            matrix[predictions[h][w]][masks[h][w]] += 1

    return matrix


def main():
    # print('hello')
    parser = argparse.ArgumentParser(prog='inference',
                                     description='Script which performs comparisons of pred and gt segmentation masks')
    parser.add_argument('--pred_mask_dirpath', required=True,
                        type=str)  # this should be FULL PATH of your images to perform inference on
    parser.add_argument('--gt_mask_dirpath', required=True, default=None,
                        type=str)  # FULL PATH of your masks to compare images with for accuracy
    parser.add_argument('--gt_mask_numclasses', required=True, default=-1,
                        type=int)  # number of classes in ground truth masks
    parser.add_argument('--output_dirpath', required=True,
                        type=str)  # this should be FULL PATH of where you want to store your predictions

    args, unknown = parser.parse_known_args()

    if args.pred_mask_dirpath is None:
        print('ERROR: missing pred_mask_dirpath')
        return

    if args.gt_mask_dirpath is None:
        print('ERROR: missing gt_mask_dirpath')
        return

    if args.gt_mask_numclasses < 0:
        print('ERROR: missing gt_mask_numclasses !!!')
        return

    print('pred_mask_dirpath:', args.pred_mask_dirpath)
    print('gt_mask_dirpath:', args.gt_mask_dirpath)
    print('gt_mask_numclasses:', args.gt_mask_numclasses)
    print('output_dirpath:', args.output_dirpath)

    if not Path(args.output_dirpath).exists():
        Path(args.output_dirpath).mkdir()

    doNamesMatch = False
    compare_batch(args.pred_mask_dirpath, args.gt_mask_dirpath, args.gt_mask_numclasses, args.output_dirpath,
                  doNamesMatch)


if __name__ == "__main__":
    main()
