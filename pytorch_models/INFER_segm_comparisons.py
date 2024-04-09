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
from sklearn.metrics import f1_score, mean_squared_error, jaccard_score, adjusted_rand_score, pair_confusion_matrix, \
    multilabel_confusion_matrix
from sklearn.metrics import _classification

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
    avg_adjusted_rand = 0.0
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
        precision_score, recall_score, accuracy_score, f1_score, jaccard_coeff, dice_coeff, mse_coeff, labelwise_dice, adjrand \
            = metrics_masks(mask_file_array[match_index], pred_mask, gt_mask, gt_mask_numclasses, output_dir)
        avg_precision_score += precision_score
        avg_recall_score += recall_score
        avg_accuracy_score += accuracy_score
        avg_f1_score += f1_score
        avg_adjusted_rand += adjrand
        ##########################
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
    batchsummary["Adjusted Rand"] = avg_adjusted_rand
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
    all_labels = sorted(list(set(np.unique(gt_mask).tolist() + np.unique(pred_mask).tolist())))
    labelwise_dice = {}
    labelwise_jaccard = jaccard_score(gt_mask.flatten(), pred_mask.flatten(), labels=all_labels, average=None)
    for l, label in enumerate(all_labels):
        labelwise_dice[f'{label}'] = (2 * labelwise_jaccard[l]) / (labelwise_jaccard[l] + 1)

    jaccard_coeff = jaccard_score(gt_mask.flatten(), pred_mask.flatten(), average='micro')
    # TODO: Adjusted Rand Index
    adjrand = adjusted_rand_score(labels_true=gt_mask.flatten(), labels_pred=pred_mask.flatten())
    dice_coeff = (2 * jaccard_coeff) / (jaccard_coeff + 1)
    mse_coeff = mean_squared_error(gt_mask.flatten(), pred_mask.flatten())
    return precision_score, recall_score, accuracy_score, f1_score, jaccard_coeff, dice_coeff, mse_coeff, labelwise_dice, adjrand


def get_pair_confusion_matrix(labels_true, labels_pred):
    CM = pair_confusion_matrix(labels_true, labels_pred)
    CM = CM.astype(np.uint16)
    return CM


def get_multilabel_confusion_matrix(labels_true, labels_pred, labelslist):
    MCM = multilabel_confusion_matrix(labels_true, labels_pred, labels=labelslist)
    return MCM


def get_tfpn(MCM):
    if MCM.ndim == 3:
        tp = MCM[:, 1, 1]
        tn = MCM[:, 0, 0]
        fp = MCM[:, 0, 1]
        fn = MCM[:, 1, 0]
    elif MCM.ndim == 2:  # single label
        tp = MCM[1, 1]
        tn = MCM[0, 0]
        fp = MCM[0, 1]
        fn = MCM[1, 0]
    else:
        raise ValueError(f"Multilabel confusion matrix must be 3 dimensional, currently f{MCM.ndim}")
    return tn, fp, fn, tp


def classification_divide(numerator, denominator, metric, modifier, average, warn_for, zero_division="warn"):
    if numerator.ndim == 2 and denominator.ndim == 2:
        _classification._prf_divide(numerator=numerator, denominator=denominator, metric=metric, modifier=modifier,
                                    average=average, warn_for=warn_for, zero_division=zero_division)


def cumulative_multilabel_jaccard(MCM):
    tn, fp, fn, tp = get_tfpn(MCM)
    numerator = tp
    denominator = tp + fp + fn
    jaccard = _classification._prf_divide(numerator, denominator, "jaccard", "true or predicted", warn_for=("jaccard",))
    return jaccard


def cumulative_multilabel_dice(MCM):
    tn, fp, fn, tp = get_tfpn(MCM)
    numerator = 2 * tp
    denominator = 2 * tp + fp + fn
    dice = _classification._prf_divide(numerator, denominator, "Dice", "true or predicted", average=None,
                                       warn_for=("Dice",))
    return dice


def cumulative_multilabel_aprf(MCM):
    tn, fp, fn, tp = get_tfpn(MCM)
    numacc = tp + tn
    denacc = tp + tn + fp + fn
    accuracy = _classification._prf_divide(numacc, denacc, "Accuracy", "true or predicted", average=None,
                                           warn_for=("Accuracy",))
    numprec = tp
    denprec = tp + fp
    precision = _classification._prf_divide(numprec, denprec, "Precision", "true or predicted", average=None,
                                            warn_for=("Precision",))
    numrec = tp
    denrec = tp + fn
    recall = _classification._prf_divide(numrec, denrec, "Recall", "true or predicted", average=None,
                                         warn_for=("Recall",))
    numf1 = 2 * precision * recall
    denf1 = precision + recall
    f1 = _classification._prf_divide(numf1, denf1, "F-1 Score", "true or predicted", average=None,
                                     warn_for=("F-1 Score",))
    return accuracy, precision, recall, f1


def cumulative_adjusted_rand(MCM):
    tn, fp, fn, tp = get_tfpn(MCM)
    numerator = (2.0 * (tp * tn - fn * fp))
    denominator = ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    dice = _classification._prf_divide(numerator, denominator, "Rand", "true or predicted", average=None,
                                       warn_for=("Rand",))
    return dice


def getMCMfromDict(labelwiseMCM):
    # It is assumed that the keys are of
    keys, values = list(labelwiseMCM.keys()), list(labelwiseMCM.values())
    keys = [int(k) for k in keys]
    labelwiseMCM = dict(sorted(labelwiseMCM.items(), key=lambda item: int(item[0])))  # .values()
    # print("LBLMCMVAL", labelwiseMCM.values())
    MCM = np.stack(list(labelwiseMCM.values()))
    return MCM, keys


def combine_multilabel_confusion_matrices(MCM_new, all_labels, labelwise_MCM=None):
    # Ensure that this function and get_multilabel_confusion_matrix() have the same value in all labels
    if labelwise_MCM is None:
        labelwise_MCM = {}
    else:
        assert isinstance(labelwise_MCM, dict), f"labelwise_MCM must be a dict, currently {type(labelwise_MCM)} "
    # MCM_count = {}
    for l, label in enumerate(all_labels):
        if label not in list(labelwise_MCM.keys()):
            labelwise_MCM[label] = 0.0
        labelwise_MCM[label] += MCM_new[l]
    return labelwise_MCM


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
