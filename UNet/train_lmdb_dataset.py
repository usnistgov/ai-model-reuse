# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import sys

from gpu_utilization import record
from gpu_utilization import write_header

if sys.version_info[0] < 3:
    print('Python3 required')
    sys.exit(1)

import os
# set the system environment so that the PCIe GPU ids match the Nvidia ids in nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
# gpus_to_use must bs comma separated list of gpu ids, e.g. "1,3,4"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0, 1" for multiple

import argparse
import datetime
import numpy as np
import unet_model
import unet_dataset
import torch
from torch import nn
import torch.cuda.amp
import time
import copy
import csv
#import pandas as pd
""" DeepLabv3 Model download and change the head for your prediction"""


def confusion_matrix(predictions, masks, number_classes):
    matrix = np.zeros((number_classes, number_classes))
    for i in range(predictions.shape[0]):
        pred = torch.argmax(predictions[i], 0)
        pred = pred.cpu().detach().numpy().astype(np.uint8)
        #mask = torch.squeeze(masks[i], 0)
        mask = masks[i]
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
        #mask = torch.squeeze(masks[i], 0)
        mask = masks[i]
        mask = mask.cpu().detach().numpy().astype(np.uint8)
        matched = np.sum(pred == mask)
        total_acc += (matched /(pred.shape[0] * pred.shape[1]))
    avg_acc = total_acc / length
    return avg_acc


def train_model(output_folder, batch_size, reader_count, train_lmdb_filepath, test_lmdb_filepath, use_augmentation, mn, mfn, cfname, number_classes, balance_classes, learning_rate, test_every_n_steps, number_of_epochs, use_amp: bool = True):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    use_gpu = torch.cuda.is_available()
    print('is gpu available',use_gpu)
    num_workers = 1 #int(config['batch_size'] / 2)
    # torch_model_ofp = os.path.join(output_folder, 'unet_checkpoint')
    # if os.path.exists(torch_model_ofp):
    #     import shutil
    #     shutil.rmtree(torch_model_ofp)
    # os.makedirs(torch_model_ofp)

    print('batch_size',batch_size)
    pin_dataloader_memory = True
    train_dataset = unet_dataset.UnetDataset(train_lmdb_filepath, number_classes, augment=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_dataloader_memory, drop_last=True)

    test_dataset = unet_dataset.UnetDataset(test_lmdb_filepath, number_classes, augment=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_dataloader_memory, drop_last=True)

    try:  # if any errors happen we want to catch them and shut down the multiprocess readers
        print('Starting Readers')

        print('Creating model')
        model = unet_model.UNet(3, number_classes) #1 channel comment change
        if use_gpu:
            model = torch.nn.DataParallel(model)
            model = model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
        train_epoch_size = test_every_n_steps
        test_epoch_size = test_dataset.get_image_count() / batch_size

        test_loss = list()


        current_time = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        #starttime = time.time()
        print('current time:', current_time)
        scaler = None
        if use_amp:
            scaler = torch.cuda.amp.GradScaler()

        epoch = 1
        epoch_loss = 0
        loss_regress = nn.SmoothL1Loss()
        criterion = torch.nn.CrossEntropyLoss()  # criterion for training (weigthed classes)
        criterion_test = torch.nn.CrossEntropyLoss()  # criterion for validation (no weighted classes)
        train_loss = list()
        test_loss_avg = list()

        # save gpu utilization
        gpu_metric_filename, info_suffix = os.path.splitext(mn)
        print('DEBUG: gpu_metric_filename:', gpu_metric_filename, ' info_suffix:', info_suffix)
        gpu_metric_filename += '_gpu.csv'
        gpu_metric_folder = os.path.join(output_folder, 'gpu_info')
        if not os.path.exists(gpu_metric_folder):
            os.makedirs(gpu_metric_folder)

        gpu_metric_filename = os.path.join(gpu_metric_folder, gpu_metric_filename)
        print('INFO: unet output gpu utilization file: ', gpu_metric_filename)
        #gpu_utilization.write_header(gpu_metric_filename)
        write_header(gpu_metric_filename)

        fieldnames = ['Model', 'Pretrained', 'LR', 'Batch_Size','epoch', 'Seconds', 'Train_loss', 'Test_loss', 'Per-Pixel Accuracy', 'Precision', 'Recall', 'F1-Score']

        print('INFO: unet output file: ', os.path.join(output_folder, mn))

        with open(os.path.join(output_folder, mn), 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        ('Running Network')
        start_time = time.time()
        while True:  # loop until if epoch >= number_of_epochs:
            model.train()  # put the model in training mode
            batch_count = 0
            train_loss_running = 0
            batchsummary = {a: [0] for a in fieldnames}
            batchsummary['epoch'] = epoch
            batchsummary['Model'] = 'Unet'
            batchsummary['Pretrained'] = 'False'
            batchsummary['LR'] = learning_rate

            for i, (images, target) in enumerate(train_loader):
                target = target.type(torch.LongTensor)
                target = target.reshape([target.shape[0],images.shape[2],images.shape[3]])
                if use_gpu:
                    images = images.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)

                optimizer.zero_grad()
                batch_count = batch_count + 1
                if use_amp:
                    with torch.cuda.amp.autocast():
                        pred = model.forward(images)
                else:
                    pred = model.forward(images)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        loss = criterion(pred, target)
                else:
                    loss = criterion(pred, target)

                if np.isnan(loss.item()):
                    #print('DEBUG not a number (NaN) : pred:', pred, ' loss:', loss.item())
                    print('INFO Training part - images have identical values mapping to an identical target label: Epoch:', epoch,
                          ' batch id:', i)
                    # reset prediction and loss to 0
                    pred = torch.zeros(pred.shape, device='cuda:0', requires_grad=True)
                    loss = torch.tensor([0.], device='cuda:0', requires_grad=True)
                    # re-initialize the model
                    # https://www.cluzters.ai/forums/topic/345/how-to-initialize-weights-in-py-torch?c=1597
                    # torch.nn.init.normal_(model, mean=0, std=1)
                    # a constantdistribution              write:
                    #  torch.nn.init.constant_(tensor, value)

                    # TODO tested this option with the break being comments out
                    # the result is not convergerging since the model is re-initialized after each epoch
                    # model = unet_model.UNet(3, number_classes)  # 1 channel comment change
                    # if use_gpu:
                    #     model = torch.nn.DataParallel(model)
                    #     model = model.cuda()
                    #
                    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
                    break

                epoch_loss += np.sum(loss.item())
                sum1 = np.sum(loss.item())
                train_loss.append(sum1)
                train_loss_running += loss.item()
                print("Epoch: {} Batch {}/{} loss {}".format(epoch, i, len(train_loader), sum1))
                if use_amp:
                    # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                    # Backward passes under autocast are not recommended.
                    # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()

            # print('DEBUG: train i:', i)
            train_epoch_loss = train_loss_running / len(train_loader)
            batchsummary['Train_loss'] = train_epoch_loss
            ##############################################################
            # Test epoch
            print('running test epoch')
            test_loss = list()
            running_loss = 0
            matrices = []
            running_acc = 0
            with torch.no_grad():
                for i, (images, target) in enumerate(test_loader):
                    if i == 0:
                        print('stop')

                    target = target.type(torch.LongTensor)
                    target = target.reshape((target.shape[0],images.shape[2],images.shape[3]))
                    if use_gpu:
                        images = images.cuda(non_blocking=True)
                        target = target.cuda(non_blocking=True)
                        optimizer.zero_grad()
                        if use_amp:
                            with torch.cuda.amp.autocast():
                                pred = model.forward(images)
                        else:
                            pred = model.forward(images)

                        if use_amp:
                            with torch.cuda.amp.autocast():
                                loss = criterion_test(pred, target)
                        else:
                            loss = criterion_test(pred, target)

                        if np.isnan(loss.item()):
                            print('DEBUG not a number (NaN) : Epoch:', epoch, ' batch id:', i, ' pred:', pred, ' loss:', loss.item())
                            # print('INFO images have identical values mapping to an identical target label: Epoch:', epoch, ' batch id:', i)
                            # # reset prediction and loss to 0
                            pred = torch.zeros(pred.shape, device='cuda:0', requires_grad=True)
                            loss = torch.tensor([0.], device='cuda:0', requires_grad=True)
                            # # re-initialize the model
                            # model = unet_model.UNet(3, number_classes)  # 1 channel comment change
                            # if use_gpu:
                            #     model = torch.nn.DataParallel(model)
                            #     model = model.cuda()
                            #
                            # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

                            # https://www.cluzters.ai/forums/topic/345/how-to-initialize-weights-in-py-torch?c=1597
                            # torch.nn.init.normal_(model, mean=0, std=1)
                            #a constantdistribution              write:
                            #  torch.nn.init.constant_(tensor, value)
                            break

                        # record gpu utilization
                        if i == 0:
                            # gpu_utilization.record(epoch, gpu_metric_filename)
                            record(epoch, gpu_metric_filename)

                        sum1 = np.sum(loss.item())
                        test_loss.append(sum1)
                        running_acc += get_accuracy_batch(pred, target)
                        matrices.append(confusion_matrix(pred, target, number_classes))

            # print('DEBUG: test i:', i)

            testavg = np.mean(test_loss)
            epoch_accuracy = running_acc / len(test_loader)
            batchsummary['Per-Pixel Accuracy'] = epoch_accuracy
            batchsummary['Test_loss'] = testavg
            batchsummary['Batch_Size'] = batch_size
            cf = np.sum(matrices, 0)
            weighted_cf_row = copy.deepcopy(cf)
            running_precision = 0
            sums = np.sum(cf, axis=1).tolist()
            avgsubtract = 0
            for i in range(0, 2):
                toadd = 0
                if sums[i] == 0:
                    toadd = 0
                    avgsubtract+=1
                else:
                    toadd = cf[i][i] / sums[i]
                running_precision += toadd
            avg_precision = running_precision / (2-avgsubtract)
            # for i in range(4):
            #     for j in range(4):
            #         if sums[i] == 0:
            #             weighted_cf_row[i][j] = 0
            #         else:
            #             weighted_cf_row[i][j] /= sums[i]
            batchsummary['Precision'] = avg_precision
            col_sums = np.sum(cf, axis=0).tolist()
            running_recall = 0
            avgsubtract=0
            for i in range(0, 2):
                toadd = 0
                if col_sums[i] == 0:
                    toadd = 0
                    avgsubtract+=1
                else:
                    toadd = cf[i][i] / col_sums[i]
                running_recall += toadd
            avg_recall = running_recall / (2-avgsubtract)
            # weighted_cf_col = copy.deepcopy(cf)
            # for i in range(4):
            #     for j in range(4):
            #         if col_sums[i] == 0:
            #             weighted_cf_col[i][j] = 0
            #         else:
            #             weighted_cf_col[i][j] /= col_sums[i]
            f1 = ((avg_precision * avg_recall) / (avg_precision + avg_recall)) * 2
            batchsummary['Recall'] = avg_recall
            batchsummary['F1-Score'] = f1
            seconds = time.time() - start_time
            seconds = int(seconds)
            batchsummary['Seconds'] = seconds
            with open(os.path.join(output_folder, mn), 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(batchsummary)
            empty_row = []
            # with open(os.path.join(output_folder, cfname), 'a') as csvfile:
            #     pd.DataFrame(weighted_cf_row).to_csv(csvfile)
            #     pd.DataFrame(weighted_cf_col).to_csv(csvfile)
            #     csvfile.write('\n')


            test_loss_avg.append(testavg)
            print('this test loss',test_loss_avg)
            CONVERGENCE_TOLERANCE = 1e-4
            min_test_loss = np.min(test_loss_avg)
            error_from_best = np.abs(test_loss_avg - min_test_loss)
            error_from_best[error_from_best < CONVERGENCE_TOLERANCE] = 0
            best_epoch = np.where(error_from_best == 0)[0][0]  # unpack numpy array, select first time since that value has happened
            print('Best epoch: {}'.format(best_epoch))

            # keep this model if it is the best so far
            if (len(test_loss_avg) - 1) == np.argmin(test_loss_avg):
                #torch_model_ofp = os.path.join(output_folder, 'unet_checkpoint')
                #print("saved best unet_checkpoint in folder: ", torch_model_ofp)
                torch_model_ofp = os.path.join(output_folder, mfn)
                print("saved best unet in file: ", torch_model_ofp)
                torch.save(model, torch_model_ofp)

                # print("Saved best unet_unet_checkpoint so far in ", os.path.join(output_folder, 'unet_unet_checkpoint/{}'.format(mfn)))
                # torch.save(model, os.path.join(output_folder, 'unet_checkpoint/{}'.format(mfn)))

            # if epoch == 5 or epoch % 10 == 0:
            #     full_name = 'unet_checkpoint/same_unet_model_{}.ckpt'.format(epoch)
            #     torch.save(model, os.path.join(output_folder, full_name))
            # if len(test_loss_avg) - best_epoch > 5:
            #     break
            if epoch >= number_of_epochs:
                break
            epoch += 1


    finally: # if any errors happened during training, shut down the disk readers
        print('Shutting down train_reader')


def main():
    # Setup the Argument parsing
    parser = argparse.ArgumentParser(prog='train_unet', description='Script which trains a unet model')

    parser.add_argument('--batch_size', dest='batch_size', type=int, help='training batch size', default=4) #comment 4
    parser.add_argument('--number_classes', dest='number_classes', type=int, default=2)
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=3e-4) #comment 3e-4
    parser.add_argument('--output_dir', dest='output_folder', type=str, help='Folder where outputs will be saved (Required)', required=True)
    parser.add_argument('--test_every_n_steps', dest='test_every_n_steps', type=int, help='number of gradient update steps to take between test epochs', default=1)
    parser.add_argument('--balance_classes', dest='balance_classes', type=int, help='whether to balance classes [0 = false, 1 = true]', default=0) #set true
    parser.add_argument('--use_augmentation', dest='use_augmentation', type=int, help='whether to use data augmentation [0 = false, 1 = true]', default=1)
    parser.add_argument('--metrics_name', type=str)
    parser.add_argument('--model_filename', type=str, help='model file name', default='unet')
    parser.add_argument('--cfname', type=str)
    parser.add_argument('--train_database', dest='train_database_filepath', type=str, help='lmdb database to use for (Required)', required=True)
    parser.add_argument('--test_database', dest='test_database_filepath', type=str, help='lmdb database to use for testing (Required)', required=True)
    parser.add_argument('--number_of_epochs', dest='number_of_epochs', type=int, help='Stop training after running number_of_epochs epochs.', default=10)
    parser.add_argument('--reader_count', dest='reader_count', type=int, help='how many threads to use for disk I/O and augmentation per gpu', default=1)

    # TODO add parameter to specify the devices to use for training

    args = parser.parse_args()
    batch_size = args.batch_size
    output_folder = args.output_folder
    number_classes = args.number_classes
    number_of_epochs = args.number_of_epochs
    train_lmdb_filepath = args.train_database_filepath
    test_lmdb_filepath = args.test_database_filepath
    learning_rate = args.learning_rate
    test_every_n_steps = args.test_every_n_steps
    balance_classes = args.balance_classes
    use_augmentation = args.use_augmentation
    reader_count = args.reader_count
    mn = args.metrics_name
    mfn = args.model_filename
    cfname = args.cfname
    print('Arguments:')
    print('batch_size = {}'.format(batch_size))
    print('number_classes = {}'.format(number_classes))
    print('learning_rate = {}'.format(learning_rate))
    print('test_every_n_steps = {}'.format(test_every_n_steps))
    print('balance_classes = {}'.format(balance_classes))
    print('use_augmentation = {}'.format(use_augmentation))

    print('train_database = {}'.format(train_lmdb_filepath))
    print('test_database = {}'.format(test_lmdb_filepath))
    print('output folder = {}'.format(output_folder))

    print('number_of_epochs count = {}'.format(number_of_epochs))
    print('reader_count = {}'.format(reader_count))

    train_model(output_folder, batch_size, reader_count, train_lmdb_filepath, test_lmdb_filepath, use_augmentation, mn, mfn, cfname, number_classes, balance_classes, learning_rate, test_every_n_steps, number_of_epochs)


if __name__ == "__main__":
    main()
