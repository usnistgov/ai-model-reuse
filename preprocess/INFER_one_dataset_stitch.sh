#!/bin/bash

# example /home/pnb/trainingOutput/pytorchOutputMtoM_A10/infer_test_images/ /home/pnb/trainingOutput/pytorchOutputMtoM_A10/infer_stitch/
# source one_dataset_stitch.sh /home/pnb/trainingOutput/pytorchOutputFtoM_A10/infer_mask_images/ /home/pnb/trainingOutput/pytorchOutputFtoM_A10/infer_mask_stitch/

root_input_folder=$1
output_root_folder=$2

echo "root_input_folder: "${root_input_folder}
#echo "image_input_folder: "${image_input_folder}
#echo "mask_input_folder: "${mask_input_folder}
echo "output_root_folder: "${output_root_folder}


if [ -d "${output_root_folder}" ]; then
  # Take action if $DIR exists. #
  echo "dir exists: ${DIR}"
else
  mkdir ${output_root_folder}
fi

python INFER_stitching.py --image_dir=${root_input_folder}  --output_dir=${output_root_folder}
