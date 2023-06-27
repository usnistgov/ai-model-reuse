# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import csv
import sys
import time
import warnings
from numpy.compat import unicode
import INFER_Dataset
import numpy as np
import torch
import skimage.io
import skimage
import skimage.transform
import argparse
import os
import INFER_segm_comparisons as segm_comp
from pathlib import Path
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
from torchvision import transforms
from PIL import Image

warnings.filterwarnings("ignore")


def format_image(x):
    # reshape into tensor (CHW)
    x = np.transpose(x, [2, 0, 1])
    return x


def createDeepLabv3(outputchannels=1):
    """DeepLabv3 class with custom head
    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.
    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                    progress=True)
    model.classifier = DeepLabHead(2048, outputchannels)
    # Set the model in training mode
    model.train()
    return model


def confusion_matrix(predictions, masks, classes):
    matrix = np.zeros((classes, classes))
    for h in range(predictions.shape[0]):
        for w in range(predictions.shape[1]):
            matrix[predictions[h][w]][masks[h][w]] += 1

    # for i in range(predictions.shape[0]):
    #     pred = torch.argmax(predictions[i], 0)
    #     pred = pred.cpu().detach().numpy().astype(np.uint8)
    #     mask = torch.squeeze(masks[i], 0)
    #     mask = mask.cpu().detach().numpy().astype(np.uint8)
    #     for h in range(pred.shape[0]):
    #         for w in range(pred.shape[1]):
    #             matrix[pred[h][w]][mask[h][w]] += 1
    return matrix


''' 
run inference with model_filemath on images in image_filepath and 
save results in output_dir
'''


def inference(model_filepath, image_filepath, output_dir):
    model = torch.load(model_filepath)
    model.eval()
    file_array = []
    for filename in os.listdir(image_filepath):
        filepath = os.path.join(image_filepath, filename)
        file_array.append(filepath)

    for i in range(len(file_array)):
        image = skimage.io.imread(file_array[i])
        # image = np.expand_dims(image, 0)
        # image = np.concatenate((image, image, image), axis=0)
        if INFER_Dataset.INFERSegmentationDataset.use_normalization == "zscore_normalize":
            image = INFER_Dataset.INFERSegmentationDataset.zscore_normalize(image)
        else:
            image = image.astype(np.float32)
            pass
            # print("ignoring normalization")

        # if (image.dtype != "uint8"):
        #     image = image / 65536  # assumes uint16  --> 256*256
        #     # sample['image'] = torch.from_numpy(image)
        # else:
        #     image = image / 255
        #     # sample['image'] = sample['image'] / 255
        #     # sample['image'] = torch.from_numpy(image)

        img = torch.from_numpy(image)

        # img = Image.open(file_array[i])
        # img = img.convert('RGB')
        # data_transform = transforms.Compose([transforms.ToTensor()])
        # img = data_transform(img)
        img = torch.unsqueeze(img, 0)
        img = img.type(torch.cuda.FloatTensor)
        pred = model(img)
        pred = pred

        pred = torch.squeeze(pred, 0)
        pred = torch.argmax(pred, 0)
        pred = pred.cpu().detach().numpy().astype(np.uint8)
        # name_start = file_array[i].find('2')
        # name = file_array[i] #[name_start:100]
        basename = os.path.basename(file_array[i])
        output_fullpath = str(output_dir) + "/pred_{}".format(basename)
        # print('outputfile:', basename, end='\t')
        skimage.io.imsave(output_fullpath, pred)


'''
run inference and compute the accuracy against the ground truth mask
'''


def inference_withmask(model_filepath, image_filepath, mask_filepath, num_classes, output_dir):
    start_time = time.time()
    model = torch.load(model_filepath)
    model.eval()
    file_array = []
    for filename in os.listdir(image_filepath):
        filepath = os.path.join(image_filepath, filename)
        file_array.append(filepath)

    mask_file_array = []
    for mfilename in os.listdir(mask_filepath):
        mfilepath = os.path.join(mask_filepath, mfilename)
        mask_file_array.append(mfilepath)

    avg_precision_score = 0.0
    avg_recall_score = 0.0
    avg_accuracy_score = 0.0
    avg_f1_score = 0.0
    avg_dice = 0.0
    avg_jaccard = 0.0
    avg_mse = 0.0
    average_label_dice = {}  # empty dictionary for all labels
    all_labels = []
    for i in range(len(file_array)):
        image_basename = os.path.basename(file_array[i])
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

        # load intensity image
        image = skimage.io.imread(file_array[i])

        if INFER_Dataset.INFERSegmentationDataset.use_normalization == "zscore_normalize":
            image = INFER_Dataset.INFERSegmentationDataset.zscore_normalize(image)
        else:
            image = image.astype(np.float32)
            pass
            # print("ignoring normalization")

        img = torch.from_numpy(image)

        img = torch.unsqueeze(img, 0)
        img = img.type(torch.cuda.FloatTensor)
        pred = model(img)
        pred = pred

        pred = torch.squeeze(pred, 0)
        pred = torch.argmax(pred, 0)
        pred = pred.cpu().detach().numpy().astype(np.uint8)

        gt_mask = skimage.io.imread(mask_file_array[match_index])

        precision_score, recall_score, accuracy_score, f1_score, jaccard_score, dice_score, mse, lbl_dice = \
            segm_comp.metrics_masks(mask_file_array[match_index], pred, gt_mask, num_classes, output_dir)

        avg_precision_score += precision_score
        avg_recall_score += recall_score
        avg_accuracy_score += accuracy_score
        avg_f1_score += f1_score
        avg_dice += dice_score
        avg_jaccard += jaccard_score
        avg_mse += mse
        gt_mask_labels = np.unique(gt_mask).tolist()
        pred_labels = np.unique(pred).tolist()
        # all_labels = sorted(list(set(gt_mask_labels + pred_labels)))

        all_labels = sorted(list(set(gt_mask_labels + all_labels)))
        # print(f"\npred LABELS: {np.unique(pred)}, gt_mask LABELS: {np.unique(gt_mask)}")
        # print(f"\tALL LABELS: {all_labels}")
        # print(f"\t\tDICE LABELS: {lbl_dice}")

        for l, label in enumerate(gt_mask_labels):
            if f"Dice_{str(label)}" not in average_label_dice:
                average_label_dice[f"Dice_{str(label)}"] = 0.0
            average_label_dice[f"Dice_{str(label)}"] += lbl_dice[f"{label}"]
        ##########################
        # name_start = file_array[i].find('2')
        # name = file_array[i] #[name_start:100]

        ############
        # TODO disabled for INFER numerical evaluations - should be enabled in the future
        basename = os.path.basename(file_array[i])
        output_fullpath = str(output_dir) + "/pred_{}".format(basename)
        # print('done:', basename, end='\t')
        skimage.io.imsave(output_fullpath, pred)

    if len(file_array) > 0:
        avg_precision_score /= len(file_array)
        avg_recall_score /= len(file_array)
        avg_accuracy_score /= len(file_array)
        avg_f1_score /= len(file_array)
        avg_dice /= len(file_array)
        avg_jaccard /= len(file_array)
        avg_mse /= len(file_array)
        for l, label in enumerate(all_labels):
            if f"Dice_{str(label)}" in average_label_dice:
                average_label_dice[f"Dice_{str(label)}"] /= len(file_array)
    start_time = time.time()
    dicelist = sorted(list(average_label_dice.keys()))
    fieldnames = ['GTMask_filepath', 'PredMask_filepath', 'Precision', 'Recall', 'Accuracy', 'F1-Score', 'Dice',
                  'Jaccard', 'MSE', 'Exec_time [seconds]'] + dicelist
    metrics_name = '_metrics.csv'
    print(f"DICELIST: {dicelist}")
    path_to_file = output_dir + metrics_name
    print('INFO: summary output stats:', path_to_file)
    file_exists = os.path.exists(path_to_file)
    if not file_exists:
        with open(path_to_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    batchsummary = {a: [0] for a in fieldnames}
    batchsummary['GTMask_filepath'] = mask_filepath
    batchsummary['PredMask_filepath'] = output_dir
    batchsummary['Precision'] = avg_precision_score
    batchsummary['Recall'] = avg_recall_score
    batchsummary['Accuracy'] = avg_accuracy_score
    batchsummary['F1-Score'] = avg_f1_score
    batchsummary['Dice'] = avg_dice
    batchsummary['Jaccard'] = avg_jaccard
    batchsummary['MSE'] = avg_mse
    for l, label in enumerate(all_labels):
        if f"Dice_{str(label)}" in average_label_dice:
            if np.isnan(average_label_dice[f"Dice_{str(label)}"]) \
                    or isinstance(average_label_dice[f"Dice_{str(label)}"], list):
                average_label_dice.pop(f"Dice_{str(label)}")  # cleanup for some unintended cases
                batchsummary.pop(f"Dice_{str(label)}")  # cleanup for some unintended cases
            else:
                batchsummary[f"Dice_{str(label)}"] = average_label_dice[f"Dice_{str(label)}"]
    exec_time = time.time() - start_time
    exec_time = int(exec_time)
    batchsummary['Exec_time [seconds]'] = exec_time
    print(batchsummary)
    with open(path_to_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(batchsummary)


def main():
    # print('hello')
    parser = argparse.ArgumentParser(prog='inference',
                                     description='Script which performs inference using models in torchvision library')
    parser.add_argument('--model_filepath', required=True,
                        type=str)  # this should be FULL PATH of the weights.pt file you generated from train.py
    parser.add_argument('--image_dirpath', required=True,
                        type=str)  # this should be FULL PATH of your images to perform inference on
    parser.add_argument('--mask_dirpath', required=False, default=None,
                        type=str)  # FULL PATH of your masks to compare images with for accuracy
    parser.add_argument('--mask_numclasses', required=False, default=-1,
                        type=int)  # number of classes in ground truth masks
    parser.add_argument('--output_dirpath', required=True,
                        type=str)  # this should be FULL PATH of where you want to store your predictions

    args, unknown = parser.parse_known_args()

    if args.model_filepath is None:
        print('ERROR: missing input model_filepath')
        return

    if args.image_dirpath is None:
        print('ERROR: missing input image_dirpath')
        return

    print('model_filepath:', args.model_filepath)
    print('image_dirpath:', args.image_dirpath)
    print('mask_dirpath:', args.mask_dirpath)
    print('output_dirpath:', args.output_dirpath)

    if not os.path.exists(args.output_dirpath):
        Path(args.output_dirpath).mkdir()

    if args.mask_dirpath is None:
        inference(args.model_filepath, args.image_dirpath, args.output_dirpath)
    else:
        print('mask_dirpath:', args.mask_dirpath)
        if args.mask_numclasses < 0:
            print('ERROR: missing input mask_numclasses !!!')
            return
        print('mask_numclasses:', args.mask_numclasses)
        inference_withmask(args.model_filepath, args.image_dirpath, args.mask_dirpath, args.mask_numclasses,
                           args.output_dirpath)


if __name__ == "__main__":
    main()
