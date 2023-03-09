# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

'''
2023-03-01
this code was adapted from https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101
the code was originally designed to train AI classification models on the ImageNet training dataset

The code has been modified to train on the Kaggle PANDA dataset (H&E stained biopsies).
The code modifications required
(a) a few fixes (e.g., async was replaced with non_blocking)
(b) additions to capture training statistics
(c) expanding the arguments (e.g., specifying output directory for models and statistics)
(d) including GPU stats in the training statistics

The Kaggle dataset has been preprocessed by
(a) organizing the Kaggle data as /root_dataset/train and /root_dataset/val
(b) cropping images to 256 x 256 images in png file format
(c) renaming files to encode the class label according to TrojAI convention
train/0/class_0_example_0.png and train/1/class_1_example_0.pmg, and
val/0/cleas_0_example_0.png etc.

Training usage: main.py [-h] [--arch ARCH] [-j N] [--epochs N] [--start-epoch N] [-b N]
#                [--lr LR] [--momentum M] [--weight-decay W] [--print-freq N]
#                [--resume PATH] [-e] [--pretrained] [--output_dir PATH]
#                DIR
# main.py -arch 'alexnet' --pretrained --epochs 20 --evaluate /mnt/raid1/pnb/prostate_kaggle/test/00a7fb880dc12c5de82df39b30533da9

__author__      = "Peter Bajcsy"
__email__ = "peter.bajcsy@nist.gov"
'''
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import pandas as pd
import numpy as np


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--output_dir', dest='output_dir', default=None, type=str, metavar='OUTPUT_DIR',
                    help='output directory where the stats, and best and checkpoint models are saved')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    # prepare output directory
    if args.output_dir is None:
        print("ERROR: missing output directory")
        return
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    output_dir = args.output_dir

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            #transform.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    # setup the output csv file for saving training and evaluation information
    out_stats = 'train_val_stats.csv'
    out_csv_stats = os.path.join(output_dir, out_stats)
    train_val_label = 'train_val'
    epoch_label = 'epoch_label'
    print_iter = 'print_frequency'
    data_size = 'data size'
    batch_time_val = 'batch_time.val'
    batch_time_avg = 'batch_time_avg'
    loss_val = 'loss_val'
    loss_avg = 'loss_avg'
    top1_val = 'top1_val'
    top1_avg = 'top1_avg'
    top5_val = 'top5_val'
    top5_avg = 'top5_avg'
    data_time_val = 'data_time.val'
    data_time_avg = 'data_time_avg'
    metrics_stats = pd.DataFrame(columns=[train_val_label,epoch_label, print_iter, data_size, batch_time_val, batch_time_avg,
                                          loss_val, loss_avg,top1_val, top1_avg,top5_val, top5_avg, data_time_val, data_time_avg ])

    # create the output file only if it does not exist
    if not os.path.exists(out_csv_stats):
        metrics_stats.to_csv(out_csv_stats, mode='w', header=True, index=False)

    if args.evaluate:
        top1_avg = validate(val_loader, model, criterion, metrics_stats, out_csv_stats)
        print('INFO for validate: top1_avg = ', top1_avg)
        return

    print('INFO: args.start_epoch:', args.start_epoch)
    print('INFO: args.epochs:', args.epochs)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, metrics_stats, out_csv_stats)

        # evaluate on validation set
        prec1= validate(val_loader, model, criterion, metrics_stats, out_csv_stats)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, output_dir)

'''
the method for training a model
metrics_stats = pandas frame that contains the names of columns in the output CSV file with stats
out_csv_stats = the file name for appending information
'''
def train(train_loader, model, criterion, optimizer, epoch, metrics_stats, out_csv_stats):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    header_col = metrics_stats.columns
    #print('INFO: header_col:', header_col)

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        # losses.update(loss.data[0], input.size(0))
        # top1.update(prec1[0], input.size(0))
        # top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

            metrics_stats[header_col[0]] = ['Train']
            metrics_stats[header_col[1]] = [str(epoch)]
            metrics_stats[header_col[2]] = [str(i)]
            metrics_stats[header_col[3]] = [str(len(train_loader))]

            metrics_stats[header_col[4]] = [batch_time.val]
            metrics_stats[header_col[5]] = [batch_time.avg]
            metrics_stats[header_col[6]] = [losses.val]
            metrics_stats[header_col[7]] = [losses.avg]
            metrics_stats[header_col[8]] = [top1.val]
            metrics_stats[header_col[9]] = [top1.avg]
            metrics_stats[header_col[10]] = [top5.val]
            metrics_stats[header_col[11]] = [top5.avg]

            metrics_stats[header_col[12]] = [data_time.val]
            metrics_stats[header_col[13]] = [data_time.avg]

            metrics_stats.to_csv(out_csv_stats, mode='a', header=False, index=False)
            # print(metrics_stats)

    return


def validate(val_loader, model, criterion, metrics_stats, out_csv_stats):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()

    header_col = metrics_stats.columns
    # print('INFO: validate header_col:', header_col)

    for i, (input, target) in enumerate(val_loader):
        targetT = target.cuda(non_blocking=True)
        inputT = input.cuda(non_blocking=True)
        # input_var = torch.autograd.Variable(input, volatile=True)
        # target_var = torch.autograd.Variable(target, volatile=True)
        with torch.no_grad():
            # compute output
            output = model(inputT)
            loss = criterion(output, targetT)


        # # compute output
        # output = model(input_var)
        # loss = criterion(output, target_var)
        # print('INFO: output.data =', output.data)
        # print('INFO: target =', targetT)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, targetT, topk=(1, 5))
        print('INFO: prec1=', prec1)
        print('INFO: prec5=', prec5)

        print('INFO: loss.item()=', loss.item())
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        # losses.update(loss.data[0], input.size(0))
        # top1.update(prec1[0], input.size(0))
        # top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

            metrics_stats[header_col[0]] = ['Validate']
            metrics_stats[header_col[1]] = ['']
            metrics_stats[header_col[2]] = [str(i)]
            metrics_stats[header_col[3]] = [str(len(val_loader))]
            metrics_stats[header_col[4]] = [batch_time.val]
            metrics_stats[header_col[5]] = [batch_time.avg]
            metrics_stats[header_col[6]] = [losses.val]
            metrics_stats[header_col[7]] = [losses.avg]
            metrics_stats[header_col[8]] = [top1.val]
            metrics_stats[header_col[9]] = [top1.avg]
            metrics_stats[header_col[10]] = [top5.val]
            metrics_stats[header_col[11]] = [top5.avg]
            metrics_stats[header_col[12]] = ['']
            metrics_stats[header_col[13]] = ['']
            metrics_stats.to_csv(out_csv_stats, mode='a', header=False, index=False)

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, output_dir, filename='checkpoint.pth.tar'):
    checkpoint_filename_out = os.path.join(output_dir, filename)
    # print('INFO: save checkpoint_filename_out:', checkpoint_filename_out)
    torch.save(state, checkpoint_filename_out)
    if is_best:
        best_filename_out = os.path.join(output_dir, 'model_best.pth.tar')
        print('INFO: save best_filename_out:', best_filename_out)
        shutil.copyfile(checkpoint_filename_out, best_filename_out)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # print('INFO: pred=', pred)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # print('INFO: correct=', correct, '\n correct.size():', correct.size())
    # print('INFO: correct.dtype=', correct.dtype)
    # print('INFO: topk=', topk)
    # print('\n')

    res = []
    for k in topk:
        # print('INFO: k:', k, ' correct[:k]=',correct[:k])
        #print('INFO: correct[:k].reshape(-1)=', correct[:k].reshape(-1))
        correct_k = correct[:k].reshape(-1).float().sum(0)
        # correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
