# IEEE Format Citation:
# M. S. Minhas, “Transfer Learning for Semantic Segmentation using PyTorch DeepLab v3,” GitHub.com/msminhas93, 12-Sep-2019. [Online]. Available: https://github.com/msminhas93/DeepLabv3FineTuning.
# Link: https://towardsdatascience.com/transfer-learning-for-segmentation-using-deeplabv3-in-pytorch-f770863d6a42
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models.segmentation.lraspp import LRASPPHead, LRASPP
from torchvision.models.mobilenetv3 import MobileNetV3, _mobilenet_v3_conf

from torchvision import models
import copy
import csv
import os
import time
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
from pathlib import Path
from sklearn import metrics
from tqdm import tqdm
import datahandler
from gpu_utilization import write_header
from gpu_utilization import record

from datahandler import GetDataloader
from sklearn.metrics import f1_score, roc_auc_score
from torchvision.utils import save_image
from segdataset import SegmentationDataset


def initializeModel(outputchannels, pretrained, name):
    if name == 'Deeplab50' and pretrained is False:
        model = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=outputchannels, progress=True)
        model.train()
        return model
    if name == 'Deeplab50' and pretrained is True:
        model = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)
        model.classifier = DeepLabHead(2048, outputchannels)
        model.train()
        return model
    if name == 'Deeplab101' and pretrained is False:
        model = models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=outputchannels, progress=True)
        model.train()
        return model
    if name == 'Deeplab101' and pretrained is True:
        model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
        model.classifier = DeepLabHead(2048, outputchannels)
        model.train()
        return model
    if name == 'MobileNetV3-Large' and pretrained is False:
        model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=False, num_classes=outputchannels,
                                                                 progress=True)
        model.train()
        return model
    if name == 'MobileNetV3-Large' and pretrained is True:
        # https://gemfury.com/neilisaac/python:torchvision/-/content/models/quantization/mobilenetv3.py
        model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True, progress=True)
        print('INFO DOES NOT WORK: MobileNetV3-Large', model.parameters)
        arch = "mobilenet_v3_large"
        inverted_residual_setting1, last_channel1 = _mobilenet_v3_conf(arch)
        print('inverted_residual_setting1:', inverted_residual_setting1)
        model.classifier = MobileNetV3(inverted_residual_setting=inverted_residual_setting1, last_channel=last_channel1,
                                       num_classes=outputchannels)
        model.train()
        return model
    if name == 'LR-ASPP-MobileNetV3-Large' and pretrained is False:
        model = models.segmentation.lraspp_mobilenet_v3_large(pretrained=False, num_classes=outputchannels,
                                                              progress=True)
        model.train()
        return model
    if name == 'LR-ASPP-MobileNetV3-Large' and pretrained is True:
        # https://github.com/pytorch/vision/blob/b94a4014a68d08f37697f4672729571a46f0042d/torchvision/models/segmentation/lraspp.py
        # Lite Reduced Atrous Spatial Pyramid Pooling (LR-ASPP)
        model = models.segmentation.lraspp_mobilenet_v3_large(pretrained=True, progress=True)
        print('INFO DOES NOT WORK: LR-ASPP-MobileNetV3-Large ', model.parameters)
        model.classifier = LRASPPHead(low_channels=960, high_channels=3, num_classes=outputchannels, inter_channels=18)
        model.train()
        return model
    if name == 'Resnet50' and pretrained is False:
        model = models.segmentation.fcn_resnet50(pretrained=pretrained, num_classes=outputchannels, progress=True)
        model.train()
        return model
    if name == 'Resnet50' and pretrained is True:
        model = models.segmentation.fcn_resnet50(pretrained=pretrained, progress=True)
        model.classifier = FCNHead(2048, outputchannels)
        return model
    if name == 'Resnet101' and pretrained is False:
        model = models.segmentation.fcn_resnet101(pretrained=pretrained, num_classes=outputchannels, progress=True)
        model.train()
        return model
    if name == 'Resnet101' and pretrained is True:
        model = models.segmentation.fcn_resnet101(pretrained=pretrained, progress=True)
        model.classifier = FCNHead(2048, outputchannels)
        return model

    print('ERROR: did not find a match to the model name: ', name)
    return None


def confusion_matrix(predictions, masks, classes):
    matrix = np.zeros((classes, classes))
    for i in range(predictions.shape[0]):
        pred = torch.argmax(predictions[i], 0)
        pred = pred.cpu().detach().numpy().astype(np.uint8)
        mask = torch.squeeze(masks[i], 0)
        mask = mask.cpu().detach().numpy().astype(np.uint8)
        for h in range(pred.shape[0]):
            for w in range(pred.shape[1]):
                matrix[pred[h][w]][mask[h][w]] += 1
    return matrix


def get_accuracy_batch(predictions, masks):
    total_acc = 0
    length = predictions.shape[0]
    background = 0
    for i in range(predictions.shape[0]):
        pred = torch.argmax(predictions[i], 0)
        pred = pred.cpu().detach().numpy().astype(np.uint8)
        mask = torch.squeeze(masks[i], 0)
        mask = mask.cpu().detach().numpy().astype(np.uint8)
        matched = np.sum(pred == mask)
        total_acc += (matched / (pred.shape[0] * pred.shape[1]))
    avg_acc = total_acc / length
    return avg_acc


def train_model(model, criterion, criterion_test, dataloaders, optimizer, metrics, bpath,
                num_epochs, device_name, metrics_name, mfn, model_name, pretrained, lr, bs, classes):
    since = time.time()
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    fieldnames = ['Model', 'Pretrained', 'LR', 'Batch_Size', 'epoch', 'Seconds', 'Train_loss', 'Test_loss',
                  'Per-Pixel Accuracy', 'Precision', 'Recall', 'F1-Score']
    print('INFO: pytorch model output stats:', os.path.join(bpath, metrics_name))

    with open(os.path.join(bpath, metrics_name), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    # save gpu utilization
    gpu_metric_filename, info_suffix = os.path.splitext(metrics_name)
    print('DEBUG: gpu_metric_filename:', gpu_metric_filename, ' info_suffix:', info_suffix)
    gpu_metric_filename += '_gpu.csv'
    gpu_metric_folder = os.path.join(bpath, 'gpu_info')
    if not os.path.exists(gpu_metric_folder):
        os.makedirs(gpu_metric_folder)

    gpu_metric_filename = os.path.join(gpu_metric_folder, gpu_metric_filename)
    print('INFO: pytorch output gpu utilization file: ', gpu_metric_filename)
    # gpu_utilization.write_header(gpu_metric_filename)
    write_header(gpu_metric_filename)

    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        running_loss = 0
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        batchsummary = {a: [0] for a in fieldnames}
        batchsummary['epoch'] = epoch
        batchsummary['Model'] = model_name
        batchsummary['Pretrained'] = pretrained
        batchsummary['LR'] = lr
        batchsummary['Batch_Size'] = bs

        # set the model to a training mode
        model.train()
        for sample in tqdm(iter(dataloaders['Train'])):

            inputs = sample['image'].to(device)
            masks = sample['mask'].to(device)

            # print('\nINPUTS1:', inputs)
            # print('\nMASKS1:', masks)
            # print('\n')

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                if device_name == 'cpu':
                    inputs = inputs.type(torch.FloatTensor)
                    masks = masks.type(torch.FloatTensor)
                else:
                    inputs = inputs.type(torch.cuda.FloatTensor)
                    masks = masks.type(torch.cuda.FloatTensor)
                    # add dimension for channels = 1
                    # masks = masks.unsqueeze(1)

                # print('\nINPUTS:', inputs.shape)
                # print('\nMASKS:', masks.shape)
                # print('\n')

                outputs = model(inputs)
                if device_name == 'cpu':
                    loss = criterion(outputs['out'], masks.type(torch.LongTensor))
                else:
                    loss = criterion(outputs['out'], masks.type(torch.cuda.LongTensor))
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
        epoch_loss = running_loss / len(dataloaders['Train'])
        batchsummary['Train_loss'] = epoch_loss

        # record gpu utilization
        # gpu_utilization.record(epoch, gpu_metric_filename)
        record(epoch, gpu_metric_filename)

        # set the model into evaluation mode
        model.eval()
        running_loss = 0
        matrices = []
        running_acc = 0
        for sample in tqdm(iter(dataloaders['Test'])):
            inputs = sample['image'].to(device)
            masks = sample['mask'].to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                if device_name == 'cpu':
                    inputs = inputs.type(torch.FloatTensor)
                    masks = masks.type(torch.FloatTensor)
                else:
                    inputs = inputs.type(torch.cuda.FloatTensor)
                    masks = masks.type(torch.cuda.FloatTensor)
                outputs = model(inputs)
                running_acc += get_accuracy_batch(outputs['out'], masks)
                matrices.append(confusion_matrix(outputs['out'], masks, classes))
                masks = masks.squeeze(1)
                if device_name == 'cpu':
                    test_loss = criterion_test(outputs['out'], masks.type(torch.LongTensor))
                else:
                    test_loss = criterion_test(outputs['out'], masks.type(torch.cuda.LongTensor))
                running_loss += test_loss.item()
                if test_loss < best_loss:
                    best_loss = test_loss
        epoch_loss = running_loss / len(dataloaders['Test'])
        epoch_accuracy = running_acc / len(dataloaders['Test'])
        cf = np.sum(matrices, 0)
        running_precision = 0
        sums = np.sum(cf, axis=1).tolist()
        avgsubtract = 0
        for i in range(0, classes):
            toadd = 0
            if sums[i] == 0:
                toadd = 0
                avgsubtract += 1
            else:
                toadd = cf[i][i] / sums[i]
            running_precision += toadd
        avg_precision = running_precision / (classes - avgsubtract)
        col_sums = np.sum(cf, axis=0).tolist()
        running_recall = 0
        avgsubtract = 0
        for i in range(0, classes):
            toadd = 0
            if col_sums[i] == 0:
                toadd = 0
                avgsubtract += 1
            else:
                toadd = cf[i][i] / col_sums[i]
            running_recall += toadd
        avg_recall = running_recall / (classes - avgsubtract)
        f1 = ((avg_precision * avg_recall) / (avg_precision + avg_recall)) * 2
        batchsummary['Per-Pixel Accuracy'] = epoch_accuracy
        batchsummary['Test_loss'] = epoch_loss
        batchsummary['Precision'] = avg_precision
        batchsummary['Recall'] = avg_recall
        batchsummary['F1-Score'] = f1
        seconds = time.time() - start_time
        seconds = int(seconds)
        batchsummary['Seconds'] = seconds
        print(batchsummary)
        with open(os.path.join(bpath, metrics_name), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
        if epoch == num_epochs:
            fullPath = str(bpath) + '/' + str(mfn)
            torch.save(model, fullPath)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))


def main():
    parser = argparse.ArgumentParser(prog='inference', description='Script which trains a Deeplabv3 model')
    parser.add_argument('--data', type=str)
    parser.add_argument('--train_images', type=str)
    parser.add_argument('--train_masks', type=str)
    parser.add_argument('--test_images', type=str)
    parser.add_argument('--test_masks', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--model_filename', type=str)
    parser.add_argument('--device_name', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--metrics_name', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--pretrained')
    parser.add_argument('--classes', type=int)

    args, unknown = parser.parse_known_args()

    if args.data is None:
        print('ERROR: missing input data')
        return
    """
            Args:
            `   **ALL ARGUMENTS REQUIRED**
            
                data (str): This is the FULL PATH of the directory containing the images and masks. 
                Structure:   data -----> set this path as the data argument
                                images
                                    image1
                                    image2
                                masks
                                    mask1
                                    mask2
                
                (train)(test)_images (str): NAME of the folder that contains the images in the data directory. 
                Structure:   data
                                images -----> set this folder name as the images argument
                                    image1
                                    image2
                                masks
                                    mask1
                                    mask2
                
                (train)(test)_masks (str): NAME of the folder that contains the masks in the data directory. 
                Structure:   data
                                images
                                    image1
                                    image2
                                masks -----> set this folder name as the masks argument
                                    mask1
                                    mask2
                
                output_dir (str): FULL PATH of directory where you want output to be stored. 2 things will be stored here: the model weights, 
                and the log.csv file containing the metrics
                
                epochs (int): number of epochs to be trained
                
                model_filename (str): This is the NAME of the .pt file where the model's weights will be stored. This will be located in your
                output_dir.  Example: "weights.pt"
                
                device_name (str): should be set to "gpu" or "cpu"
                
                batch_size (int) : batch size
                
                learning_rate (float) : learning rate
                
                metrics_name (str) : name of the csv file that metrics will be stored. Example: "metrics.csv"
                
                model_name (str) : name of the model you want (i.e Deeplab, Resnet)
                
                pretrained (str) : either "True" or "False" to obtain a pretrained pytorch model or train from scratch.  This string will be converted to a Python bool
                
                classes (int) : number of classes of the dataset. Background is counted as a class
                
                
                
            
            """

    if args.device_name == 'cpu':
        data = args.data
    else:
        data = args.data

    dataloader_class = GetDataloader(data_dir=data,  # create class that contains the dataloaders and class weights
                                     train_image_folder=args.train_images,
                                     train_mask_folder=args.train_masks,
                                     test_image_folder=args.test_images,
                                     test_mask_folder=args.test_masks,
                                     batch_size=args.batch_size,
                                     fraction=0.2,
                                     n_classes=args.classes)

    seg_dataloader = dataloader_class.dataloaders  # grab dataloader from the class
    if (len(seg_dataloader['Train']) < 1 or len(seg_dataloader['Test']) < 1):
        print('ERROR: could not find train or test data len(train):', (seg_dataloader['Train']))
        print('len(test):', len(seg_dataloader['Test']))

    toBool = True
    if args.pretrained == 'False':
        toBool = False
    model = initializeModel(outputchannels=args.classes, pretrained=toBool,
                            name=args.model_name)  # Calls function to create the deeplabv3 model from torchvision.
    if args.device_name == 'cpu':
        output_dir = args.output_dir
    else:
        output_dir = args.output_dir

    if not Path(output_dir).exists():
        Path(output_dir).mkdir()

    my_weights = dataloader_class.weights
    total_pixels = sum(my_weights)
    if total_pixels < 1:
        print("ERROR: total_pixels < 1 : {}".format(total_pixels))
        return

    fractions = []
    for number in my_weights:
        fractions.append(1.0 - (number / total_pixels))

    if args.device_name == 'cpu':
        my_weights = torch.FloatTensor(fractions)
    else:
        my_weights = torch.cuda.FloatTensor(fractions)
    print("Weighted Classes: {}".format(my_weights))
    criterion = torch.nn.CrossEntropyLoss(weight=my_weights)  # criterion for training (weigthed classes)
    criterion_test = torch.nn.CrossEntropyLoss()  # criterion for validation (no weighted classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}

    print('Starting training now')
    train_model(model, criterion, criterion_test, seg_dataloader, optimizer, bpath=output_dir, metrics=metrics,
                num_epochs=args.epochs, device_name=args.device_name, metrics_name=args.metrics_name,
                mfn=args.model_filename, model_name=args.model_name, pretrained=args.pretrained, lr=args.learning_rate,
                bs=args.batch_size, classes=args.classes)


if __name__ == "__main__":
    main()
