#!/bin/bash

# example argument: /home/pnb/trainingOutput/pytorchOutput_A10/
root_folder=$1

echo "root_folder: "${root_folder}
#echo "image_input_folder: "${image_input_folder}
#echo "mask_input_folder: "${mask_input_folder}

# --input_dir /home/pnb/trainingOutput/pytorchOutput_A10 --output_dir /home/pnb/trainingOutput/pytorchOutput_A10/graphs
python power_function_fit.py --input_dir=${root_folder} --output_dir=${root_folder}"/graphs"

# --input_dir /home/pnb/trainingOutput/pytorchOutput_A10/graphs --output_dir /home/pnb/trainingOutput/pytorchOutput_A10/comparisons
python comparison_metrics.py --input_dir=${root_folder}"/graphs" --output_dir=${root_folder}"/comparisons"

#--input_dir /home/pnb/trainingOutput/pytorchOutput_concrete/gpu_info/ --output_dir /home/pnb/trainingOutput/pytorchOutput_concrete/comparisons/
python gpu_metrics.py --input_dir=${root_folder}"/gpu_info" --output_dir=${root_folder}"/comparisons"

python plot_ranking.py --input_dir=${root_folder}"/comparisons" --output_dir=${root_folder}"/comparisons"

