#!/bin/bash

# example /home/pnb/trainingData/A10 images binary_masks 2 2
# example /home/pnb/trainingData/cryoem images masks 8 8
# example /home/pnb/trainingData/infer14 images masks 6 6

root_folder=$1
image_input_folder=$2
mask_input_folder=$3
xPieces=$4
yPieces=$5

echo "root_folder: "${root_folder}
echo "image_input_folder: "${image_input_folder}
echo "mask_input_folder: "${mask_input_folder}
echo "xPieces: "${xPieces}
echo "yPieces: "${yPieces}

python tiling.py --image_dir=${root_folder}"/"${image_input_folder}  --output_dir=${root_folder}"/tiled_images" --xPieces=${xPieces} --yPieces=${yPieces}

python tiling.py --image_dir=${root_folder}"/"${mask_input_folder}  --output_dir=${root_folder}"/tiled_masks" --xPieces=${xPieces} --yPieces=${yPieces}

python split.py --image_dir=${root_folder}"/"${ch}"/tiled_images" --mask_dir=${root_folder}"/"${ch}"/tiled_masks" --trainImageDir=${root_folder}"/"${ch}"/train_images" --trainMaskDir=${root_folder}"/"${ch}"/train_masks" --testImageDir=${root_folder}"/"${ch}"/test_images" --testMaskDir=${root_folder}"/"${ch}"/test_masks" --fraction 0.8

