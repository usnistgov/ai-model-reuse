# Preprocessing Kaggle H&E stained biopsy images and training AI classification models

## Goals
1. preprocess Kaggle H & E stained biopsy images and masks into tiles with assigned class labels (Gleason score)
2. compute statistics of preprocessed (tiled, classified, renamed) images to enable sampling
3. prepare a class-balanced training dataset 
4. train AI classification model on the trainig data
5. compare AI model statistics

## Kaggle PANDA dataseet
- Name: the cANcer graDe Assessment (PANDA) Challenge 
- Purpose: Prostate cancer diagnosis using the Gleason grading system. 
- Description: It is an image classification challenge that was summarized in 
the 2022 publication posted at [URL](https://www.nature.com/articles/s41591-021-01620-2#MOESM1)
- Dataset origin: The challenge consisted data pairs (image, image mask with pixel labels) 
coming from 6 contributing sites [URL](https://www.nature.com/articles/s41591-021-01620-2/tables/1)
- Image specifications (one image)
-- Pixel size: 33K x 5K pixels
-- 21 135 files
-- 411.9 GB

## 1. preprocess Kaggle H & E stained biopsy images and masks into tiles with assigned class labels (Gleason score)
Create tiles of size xDim x yDim and derive Gleason score per tile based on the highest mask score. 
The output tile names follow the naming convention class_GleasonScore_rowID_colID.png and the tiles are placed in 
subdirectories denoting the Gleason score.

`python tiling_fixed_size.py 
--image_dir /home/pnb/raid1/prostate_kaggle/selected_images
--mask_dir /home/pnb/raid1/prostate_kaggle/selected_masks
--output_dir /home/pnb/raid1/prostate_kaggle/test
--xDim 256
--yDim 256`

## 2. compute statistics of preprocessed (tiled, classified, renamed) images to enable sampling
Compute a histogram of the number of example images per image class 
from a set of folders with subfolder names indicating the class label.
Save the number of tile images per class label in each folder corresponding to one big image.

`python lael_class_distribution.py 
--input_dir
/mnt/raid1/pnb/prostate_kaggle/class_tiled
--output_dir
/mnt/raid1/pnb/prostate_kaggle/test`

## 3. prepare a class-balanced training dataset 
Sample images for the same class labels from subfolders with tiles of folders corresponding to big images.
Split those sampled images into two folders (train and val) according to the split fraction.

`python lael_class_split.py 
--input_dir /mnt/raid1/pnb/prostate_kaggle/class_tiled/
--output_dir /mnt/raid1/pnb/prostate_kaggle/classification_model
--fraction 0.8
--num_samples_class 3`

## 4. train AI classification model on the prepared training data
Train AI model architecture following the usage below.
- `usage: main.py [-h] [--arch ARCH] [-j N] [--epochs N] [--start-epoch N] [-b N]
                [--lr LR] [--momentum M] [--weight-decay W] [--print-freq N]
                [--resume PATH] [-e] [--pretrained] [--output_dir PATH]
                DIR`
- DIR: is the directory with training data with sub-folders train and val
- Default hyper-parameters: 
    optimizer = stochastic gradient descent (optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay))
    epochs (epochs) = 90
    batch_size  (b) = 256, 
    learning rate (lr) = 0.1,
    momentum during optimization (momentum) = 0.9, 
    weight decay during optimization (weight-decay) = 1e-4 
    
`python main.py 
--arch resnet18
--output_dir /mnt/raid1/pnb/prostate_kaggle/test/model_out
--epochs 2
/mnt/raid1/pnb/prostate_kaggle/test/00a7fb880dc12c5de82df39b30533da9/`

## 5. compare AI model statistics
Work in progress
