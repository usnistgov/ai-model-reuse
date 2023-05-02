#!/bin/bash

model_directory=$1
tiled_image_directory=$2
tiled_mask_directory=$3

#echo 'name_dataset: '$name_dataset
echo 'model_directory: '$model_directory
echo 'tiled_image_directory: '$tiled_image_directory

infer_output_directory=${model_directory}'/infer_tile_images/'
stitch_output_directory=${model_directory}'/infer_stitch/'
echo 'infer_output_directory: '$infer_output_directory
echo 'stitch_output_directory: '$stitch_output_directory

source inference_many_models_INFER.sh ${model_directory} ${tiled_image_directory} ${tiled_mask_directory} ${infer_output_directory}

cd preprocess
source INFER_one_dataset_stitch.sh ${infer_output_directory} ${stitch_output_directory}
cd ..
