# IEEE Format Citation:
# M. S. Minhas, “Transfer Learning for Semantic Segmentation using PyTorch DeepLab v3,” GitHub.com/msminhas93, 12-Sep-2019. [Online]. Available: https://github.com/msminhas93/DeepLabv3FineTuning.
# Link: https://towardsdatascience.com/transfer-learning-for-segmentation-using-deeplabv3-in-pytorch-f770863d6a42
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models.segmentation.lraspp import LRASPPHead
from torchvision.models.mobilenetv3 import MobileNetV3, _mobilenet_v3_conf
from torchvision import models
from INFER_input_model import *
from metrics import *
import csv
import os
import time
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from gpu_utilization import write_header
from gpu_utilization import record
from INFER_Dataset import GetDataloader


def initializeModel(output_channels, pretrained, name, input_channels=252, bs=80, windowsize=200):
    model = None
    combine = True
    if name == 'None':
        model = None
    if name == 'LSTM_Default':
        model = LSTMModel(input_size=input_channels, hidden_size=128, num_layers=2, num_classes=output_channels)
        combine = False
    if name == 'GRU_Tomo':
        model = GRU_3D_Model(input_size=input_channels, hidden_size=128, num_layers=2, num_classes=output_channels)
        combine = False
    if name == 'GRU_2D':
        model = GRUModel(input_size=input_channels, hidden_size=128, num_layers=2, num_classes=output_channels)
        combine = False
    if name == 'Conv1d_Default':
        model = INFERFeatureExtractor1D(input_channels=input_channels, standalone=True, output_channels=output_channels)
        combine = False
    if name == 'Deeplab50' and pretrained is False:
        model = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=output_channels, progress=True)
    if name == 'Deeplab50' and pretrained is True:
        model = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)
        model.classifier = DeepLabHead(2048, output_channels)
        model.train()
    if name == 'Deeplab101' and pretrained is False:
        model = models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=output_channels, progress=True)
    if name == 'Deeplab101' and pretrained is True:
        model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
        model.classifier = DeepLabHead(2048, output_channels)
        model.train()
    if name == 'MobileNetV3-Large' and pretrained is False:
        model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=False, num_classes=output_channels,
                                                                 progress=True)
    if name == 'MobileNetV3-Large' and pretrained is True:
        # https://gemfury.com/neilisaac/python:torchvision/-/content/models/quantization/mobilenetv3.py
        model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True, progress=True)
        print('INFO DOES NOT WORK: MobileNetV3-Large', model.parameters)
        arch = "mobilenet_v3_large"
        inverted_residual_setting1, last_channel1 = _mobilenet_v3_conf(arch)
        print('inverted_residual_setting1:', inverted_residual_setting1)
        model.classifier = MobileNetV3(inverted_residual_setting=inverted_residual_setting1, last_channel=last_channel1,
                                       num_classes=output_channels)
        model.train()
    if name == 'LR-ASPP-MobileNetV3-Large' and pretrained is False:
        model = models.segmentation.lraspp_mobilenet_v3_large(pretrained=False, num_classes=output_channels,
                                                              progress=True)
    if name == 'LR-ASPP-MobileNetV3-Large' and pretrained is True:
        # https://github.com/pytorch/vision/blob/b94a4014a68d08f37697f4672729571a46f0042d/torchvision/models/segmentation/lraspp.py
        # Lite Reduced Atrous Spatial Pyramid Pooling (LR-ASPP)
        model = models.segmentation.lraspp_mobilenet_v3_large(pretrained=True, progress=True)
        print('INFO DOES NOT WORK: LR-ASPP-MobileNetV3-Large ', model.parameters)
        model.classifier = LRASPPHead(low_channels=960, high_channels=3, num_classes=output_channels, inter_channels=18)
        model.train()
    if name == 'Resnet50' and pretrained is False:
        model = models.segmentation.fcn_resnet50(pretrained=pretrained, num_classes=output_channels, progress=True)
    if name == 'Resnet50' and pretrained is True:
        model = models.segmentation.fcn_resnet50(pretrained=pretrained, progress=True)
        model.classifier = FCNHead(2048, output_channels)
    if name == 'Resnet101' and pretrained is False:
        model = models.segmentation.fcn_resnet101(pretrained=pretrained, num_classes=output_channels, progress=True)
    if name == 'Resnet101' and pretrained is True:
        model = models.segmentation.fcn_resnet101(pretrained=pretrained, progress=True)
        model.classifier = FCNHead(2048, output_channels)

    if name is None:
        print(f'Did not find a match to the model name: {name}, using Basic Feature Extractor')
    if combine:
        selected_model = CombinedModel(input_shape=input_channels, segmentation_model=model, n_classes=output_channels,
                                       window_size=windowsize, batchsize=bs)
    else:
        selected_model = model
    if not pretrained:
        selected_model.train()
    return selected_model  # Will be none if model is not initialized


def train_model(model, criterion, criterion_test, dataloaders, optimizer, bpath, num_epochs, devicetype, metricsfile,
                mfn, modelName, pretrained, lr, bs, classes):
    since = time.time()
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    min_dice = 0
    # best_epoch = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    # TODO: add chosen_metrics keys to fieldnames
    # fieldnames = ['Model', 'Pretrained', 'LR', 'Batch_Size', 'epoch', 'Seconds', 'Train_loss', 'Test_loss',
    #               'Per-Pixel Accuracy', 'Precision', 'Recall', 'F1-Score', 'Dice', 'Jaccard', 'MSE']
    fieldnames = ['Model', 'Pretrained', 'LR', 'Batch_Size', 'epoch', 'Seconds', 'Train_loss', 'Test_loss',
                  'Per-Pixel Accuracy', 'Precision_macro', 'Precision_micro', 'Recall_macro', 'Recall_micro',
                  'Dice_macro', 'Dice_micro', 'Jaccard_macro', 'Jaccard_micro', 'confidence_sd_macro',
                  'confidence_sd_micro', 'MSE']

    print('INFO: pytorch model output stats:', os.path.join(bpath, metricsfile))

    with open(os.path.join(bpath, metricsfile), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    # save gpu utilization
    gpu_metric_filename, info_suffix = os.path.splitext(metricsfile)
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
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)
        batchsummary = {a: [0] for a in fieldnames}
        batchsummary['epoch'] = epoch
        batchsummary['Model'] = modelName
        batchsummary['Pretrained'] = pretrained
        batchsummary['LR'] = lr
        batchsummary['Batch_Size'] = bs

        # set the model to a training mode
        model.train()
        for sample in tqdm(iter(dataloaders['Train'])):

            inputs = sample['image'].to(device)
            masks = sample['mask'].to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                if devicetype == 'cpu':
                    inputs = inputs.type(torch.FloatTensor)
                    masks = masks.type(torch.FloatTensor)
                else:
                    inputs = inputs.type(torch.cuda.FloatTensor)
                    masks = masks.type(torch.cuda.FloatTensor)
                    # add dimension for channels = 1
                    # masks = masks.unsqueeze(1)
                outputs = model(inputs)
                if devicetype == 'cpu':
                    loss = criterion(outputs, masks.type(torch.LongTensor))
                else:
                    loss = criterion(outputs, masks.type(torch.cuda.LongTensor))
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
        epoch_loss = running_loss / len(dataloaders['Train'])
        batchsummary['Train_loss'] = epoch_loss

        # record gpu utilization
        record(epoch, gpu_metric_filename)

        # set the model into evaluation mode
        model.eval()
        matrices = []
        running_loss = 0
        running_acc = 0
        running_dice = 0
        running_jaccard = 0
        running_mse = 0
        for sample in tqdm(iter(dataloaders['Test'])):
            inputs = sample['image'].to(device)
            masks = sample['mask'].to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                if devicetype == 'cpu':
                    inputs = inputs.type(torch.FloatTensor)
                    masks = masks.type(torch.FloatTensor)
                else:
                    inputs = inputs.type(torch.cuda.FloatTensor)
                    masks = masks.type(torch.cuda.FloatTensor)
                outputs = model(inputs)
                running_acc += get_accuracy_batch(outputs, masks)
                # running_dice += get_dice_batch(outputs, masks)
                # running_jaccard += get_jaccard_batch(outputs, masks)
                running_mse += get_mse_batch(outputs, masks)
                matrices.append(confusionmat(outputs, masks, classes))
                masks = masks.squeeze(1)
                if devicetype == 'cpu':
                    test_loss = criterion_test(outputs, masks.type(torch.LongTensor))
                else:
                    test_loss = criterion_test(outputs, masks.type(torch.cuda.LongTensor))
                running_loss += test_loss.item()
                if test_loss < best_loss:
                    best_loss = test_loss
                    best_epoch = epoch
                    best_model = model
        epoch_loss = running_loss / len(dataloaders['Test'])
        epoch_accuracy = running_acc / len(dataloaders['Test'])
        # epoch_dice = running_dice / len(dataloaders['Test'])
        # epoch_jaccard = running_jaccard / len(dataloaders['Test'])
        epoch_mse = running_mse / len(dataloaders['Test'])

        macro_precision, macro_recall, macro_f1, macro_jaccard, micro_precision, micro_recall, micro_f1, micro_jaccard, confidences = calculate_metrics(
            matrices, classes)
        batchsummary['Per-Pixel Accuracy'] = epoch_accuracy
        batchsummary['Test_loss'] = epoch_loss
        batchsummary['Precision_macro'] = macro_precision
        batchsummary['Precision_micro'] = micro_precision
        batchsummary['Recall_macro'] = macro_recall
        batchsummary['Recall_micro'] = micro_recall
        batchsummary['Dice_macro'] = macro_f1
        batchsummary['Dice_micro'] = micro_f1
        batchsummary['Jaccard_macro'] = macro_jaccard
        batchsummary['Jaccard_micro'] = micro_jaccard
        batchsummary['confidence_sd_micro'] = confidences['SD'][0]
        batchsummary['confidence_sd_macro'] = confidences['SD'][1]
        batchsummary['MSE'] = epoch_mse
        seconds = time.time() - start_time
        seconds = int(seconds)
        batchsummary['Seconds'] = seconds
        print(batchsummary)
        # if min_dice < epoch_dice:
        #     min_dice = epoch_dice

        with open(os.path.join(bpath, metricsfile), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
    fullPath = str(bpath) + '/' + str(mfn)
    torch.save(best_model, fullPath)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))


def main():
    parser = argparse.ArgumentParser(prog='Training', description='Script which trains a Deeplabv3 model')
    parser.add_argument('--data', type=str, help='')
    parser.add_argument('--trainImages', type=str, help='')
    parser.add_argument('--trainMasks', type=str, help='')
    parser.add_argument('--testImages', type=str, help='')
    parser.add_argument('--testMasks', type=str, help='')
    parser.add_argument('--outputDir', type=str, help='')
    parser.add_argument('--epochs', type=int, help='')
    parser.add_argument('--modelWeights', type=str, help='')
    parser.add_argument('--devicetype', type=str, help='')
    parser.add_argument('--batchsize', type=int, help='')
    parser.add_argument('--learningRate', type=float, help='')
    parser.add_argument('--metricsfile', type=str, help='')
    parser.add_argument('--modelName', type=str, help='')
    parser.add_argument('--pretrained', type=bool, help='')
    parser.add_argument('--classes', type=int, help='')
    parser.add_argument('--inputchannels', type=int, help='')
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

                outputDir (str): FULL PATH of directory where you want output to be stored. 2 things will be stored here: the model weights, 
                and the log.csv file containing the metrics

                epochs (int): number of epochs to be trained

                modelWeights (str): This is the NAME of the .pt file where the model's weights will be stored. This will be located in your
                outputDir.  Example: "weights.pt"

                devicetype (str): should be set to "gpu" or "cpu"

                batchsize (int) : batch size

                learningRate (float) : learning rate

                metricsfile (str) : name of the csv file that metrics will be stored. Example: "metrics.csv"

                modelName (str) : name of the model you want (i.e Deeplab, Resnet)

                pretrained (str) : either "True" or "False" to obtain a pretrained pytorch model or train from scratch.  This string will be converted to a Python bool

                classes (int) : number of classes of the dataset. Background is counted as a class




            """

    data = args.data
    print("args", args)
    # trainImages = None
    # trainMasks = None
    # testImages = None
    # testMasks = None
    try:
        if os.path.exists(args.trainImages) and os.path.exists(args.trainMasks) and os.path.exists(
                args.testImages) and os.path.exists(args.testMasks):
            trainImages = args.trainImages
            trainMasks = args.trainMasks
            testImages = args.testImages
            testMasks = args.testMasks
    except:
        trainImages = os.path.join(data, args.trainImages)
        trainMasks = os.path.join(data, args.trainMasks)
        testImages = os.path.join(data, args.testImages)
        testMasks = os.path.join(data, args.testMasks)
    dataloader_class = GetDataloader(data_dir=data,  # create class that contains the dataloaders and class weights
                                     train_image_folder=args.trainImages,
                                     train_mask_folder=args.trainMasks,
                                     test_image_folder=args.testImages,
                                     test_mask_folder=args.testMasks,
                                     batch_size=args.batchsize,
                                     fraction=0.2,
                                     n_classes=args.classes)

    seg_dataloader = dataloader_class.dataloaders  # grab dataloader from the class
    if len(seg_dataloader['Train']) < 1 or len(seg_dataloader['Test']) < 1:
        print(trainImages, trainMasks, testImages, testMasks)
        print(args.trainImages, args.trainMasks, args.testImages, args.testMasks)
        print('ERROR: could not find train or test data len(train):', len(seg_dataloader['Train']))
        print('len(test):', len(seg_dataloader['Test']))

    pretrained = True
    if args.pretrained == 'False':
        pretrained = False
    # TODO: Add functionality for dynamically choosing windowsize
    # Calls function to create the deeplabv3 model from torchvision.
    model = initializeModel(output_channels=args.classes, pretrained=pretrained, name=args.modelName,
                            input_channels=args.inputchannels, bs=args.batchsize)
    outputDir = args.outputDir
    if not Path(outputDir).exists():
        Path(outputDir).mkdir()

    my_weights = dataloader_class.weights
    total_pixels = sum(my_weights)
    if total_pixels < 1:
        print(f"ERROR: total_pixels < 1 : {total_pixels}")
        return

    fractions = []
    for number in my_weights:
        fractions.append(1.0 - (number / total_pixels))

    if args.devicetype == 'cpu':
        my_weights = torch.FloatTensor(fractions)
    else: # Select GPU by default
        my_weights = torch.cuda.FloatTensor(fractions)
    print(f"Weighted Classes: {my_weights}")
    criterion = torch.nn.CrossEntropyLoss(weight=my_weights)  # criterion for training (weigthed classes)
    criterion_test = torch.nn.CrossEntropyLoss()  # criterion for validation (no weighted classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learningRate)

    print('Starting training now')
    print("Seg_dataloader: ", seg_dataloader, len(seg_dataloader['Train']), len(seg_dataloader['Test']))
    train_model(model, criterion, criterion_test, seg_dataloader, optimizer, bpath=outputDir, num_epochs=args.epochs,
                devicetype=args.devicetype, metricsfile=args.metricsfile, mfn=args.modelWeights,
                modelName=args.modelName, pretrained=args.pretrained, lr=args.learningRate, bs=args.batchsize,
                classes=args.classes)


if __name__ == "__main__":
    main()
