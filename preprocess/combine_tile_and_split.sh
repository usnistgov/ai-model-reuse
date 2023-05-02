#!/bin/bash

# example /home/pnb/trainingData/A10 images binary_masks 2 2
# example /home/pnb/trainingData/cryoem images masks 8 8
# example /home/pnb/trainingData/infer14 images masks 6 6
# shellcheck disable=SC1090
# shellcheck disable=SC2088
source ~/envs/airec_test/bin/activate
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

#inputs_maindir=${root_folder}"/"${image_input_folder}
python tiling.py --image_dir=${root_folder}"/"${mask_input_folder} --output_dir=${root_folder}"/tiled_masks" --xPieces=${xPieces} --yPieces=${yPieces}
python combine_and_tile.py --image_dir=${root_folder}"/"${image_input_folder} --output_dir=${root_folder}"/tiled_images" --channels "H1" "H1dark" --xPieces=${xPieces} --yPieces=${yPieces}

python split.py --image_dir=${root_folder}"/tiled_images" --mask_dir=${root_folder}"/tiled_masks" --train_image_dir=${root_folder}"/train_images" --train_mask_dir=${root_folder}"/train_masks" --test_image_dir=${root_folder}"/test_images" --test_mask_dir=${root_folder}"/test_masks" --fraction 0.8
