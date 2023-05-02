#!/bin/bash

# example /home/pnb/trainingData/A10 images binary_masks 2 2
# example /home/pnb/trainingData/cryoem images masks 8 8
# example /home/pnb/trainingData/infer14 images masks 6 6
# shellcheck disable=SC1090
# shellcheck disable=SC2088
source ~/envs/airec_test/bin/activate
root_folder=$1
image_input_folder=$2
xPieces=$3
yPieces=$4

echo "root_folder: "${root_folder}
echo "image_input_folder: "${image_input_folder}
echo "xPieces: "${xPieces}
echo "yPieces: "${yPieces}

#inputs_maindir=${root_folder}"/"${image_input_folder}
python combine_and_tile.py --image_dir=${root_folder}"/"${image_input_folder} --output_dir=${root_folder}"/tiled_"${image_input_folder} --channels "H1dark" --xPieces=${xPieces} --yPieces=${yPieces}
