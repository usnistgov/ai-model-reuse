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
zPieces=$6
channels=$7
echo "$#" arguments

ch=$7
if [ "$#" -eq 8  ]; then
  channels+=" $8"
  ch+="$8"
fi
echo channels=$channels
echo ch=$ch

echo "root_folder: "${root_folder}
echo "image_input_folder: "${image_input_folder}
echo "mask_input_folder: "${mask_input_folder}
echo "xPieces: "${xPieces}
echo "yPieces: "${yPieces}
echo "zPieces: "${zPieces}

if [ ! -d ${root_folder}"/"${ch} ]; then
	mkdir ${root_folder}"/"${ch}
  fi


#inputs_maindir=${root_folder}"/"${image_input_folder}
#python combine_and_tile.py --image_dir=${root_folder}"/"${image_input_folder} --output_dir=${root_folder}"/"${ch}"/tiled_images" --channels $channels --xPieces=${xPieces} --yPieces=${yPieces} --zPieces=${zPieces}

python tiling.py --image_dir=${root_folder}"/"${image_input_folder} --output_dir=${root_folder}"/"${ch}"/tiled_images" --xPieces=${xPieces} --yPieces=${yPieces} --zPieces=${zPieces}
python tiling.py --image_dir=${root_folder}"/"${mask_input_folder} --output_dir=${root_folder}"/"${ch}"/tiled_masks" --xPieces=${xPieces} --yPieces=${yPieces} --zPieces=${zPieces}
python split.py --image_dir=${root_folder}"/"${ch}"/tiled_images" --mask_dir=${root_folder}"/"${ch}"/tiled_masks" --trainImageDir=${root_folder}"/"${ch}"/train_images" --trainMaskDir=${root_folder}"/"${ch}"/train_masks" --testImageDir=${root_folder}"/"${ch}"/test_images" --testMaskDir=${root_folder}"/"${ch}"/test_masks" --fraction 0.8
