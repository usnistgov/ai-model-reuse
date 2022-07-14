#!/bin/bash

# example arguments: /home/pnb/trainingOutput/pytorchOutputFtoM_A10/ /home/pnb/trainingData/A10/train_images/ /home/pnb/trainingData/A10/test_images/
model_directory=$1
# /home/pnb/trainingData/A10/train_images/
train_image_directory=$2
# /home/pnb/trainingData/A10/test_images/
test_image_directory=$3
# /home/pnb/trainingOutput/pytorchOutputFtoM_A10/infer_tile_images/
#infer_output_directory=$4
# /home/pnb/trainingOutput/pytorchOutputFtoM_A10/infer_stitch/
#stitch_output_directory=$5


#echo 'name_dataset: '$name_dataset
echo 'model_directory: '$model_directory
echo 'train_image_directory: '$train_image_directory
echo 'test_image_directory: '$test_image_directory

infer_output_directory=${model_directory}'/infer_tile_images/'
stitch_output_directory=${model_directory}'/infer_stitch/'
echo 'infer_output_directory: '$infer_output_directory
echo 'stitch_output_directory: '$stitch_output_directory


source inference_many_models.sh ${model_directory} ${test_image_directory} ${infer_output_directory}

source inference_many_models.sh ${model_directory} ${train_image_directory} ${infer_output_directory}

cd preprocess
source one_dataset_stitch.sh ${infer_output_directory} ${stitch_output_directory}
cd ..

#source inference_many_models.sh /home/pnb/trainingOutput/pytorchOutputFtoM_A10/ /home/pnb/trainingData/A10/test_images/ /home/pnb/trainingOutput/pytorchOutputFtoM_A10/infer_tile_images/
#
#source inference_many_models.sh /home/pnb/trainingOutput/pytorchOutputFtoM_A10/ /home/pnb/trainingData/A10/train_images/ /home/pnb/trainingOutput/pytorchOutputFtoM_A10/infer_tile_images/
#
#source one_dataset_stitch.sh /home/pnb/trainingOutput/pytorchOutputFtoM_A10/infer_tile_images/ /home/pnb/trainingOutput/pytorchOutputFtoM_A10/infer_stitch/


