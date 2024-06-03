#!/bin/bash

model_directory=$1
tiled_image_directory=$2
tiled_mask_directory=$3
infer_output_directory_name="inference_opposite_Evaluated"
stitch_output_directory_name="stitched_Evaluated"

echo 'model_directory: '$model_directory
echo 'tiled_image_directory: '$tiled_image_directory

infer_output_directory=${model_directory}'/'${infer_output_directory_name}
stitch_output_directory=${model_directory}'/'${stitch_output_directory_name}
echo 'infer_output_directory: '$infer_output_directory
echo 'stitch_output_directory: '$stitch_output_directory
if [ ! -d infer_output_directory ]; then
  #echo 'create a folder :'$output_dir
  mkdir -p infer_output_directory
fi
if [ ! -d $stitch_output_directory ]; then
  #echo 'create a folder :'$output_dir
  mkdir -p $stitch_output_directory
fi

source INFER_inference_many_models.sh ${model_directory} ${tiled_image_directory} ${tiled_mask_directory} ${infer_output_directory}

cd preprocess
source INFER_one_dataset_stitch.sh ${infer_output_directory} ${stitch_output_directory}
cd ..
