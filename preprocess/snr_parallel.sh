#!/bin/bash

root_folder=$1
output_folder=$2

for name_dataset in cryoem A10 OCI cardiff rpe2d rpe3d concrete infer14
do
  #sbatch ./evaluate_models_round7_sbatch.sh $nS $nD $pm $lp
  image_input_folder=$root_folder"/"${name_dataset}"/images"
  mask_input_folder=$root_folder"/"${name_dataset}"/masks"
  echo $image_input_folder
  echo $mask_input_folder
  python ./snr_image.py --image_dir $image_input_folder --mask_dir $mask_input_folder --output_dir $output_folder --name_dataset $name_dataset
done
