#!/bin/bash

# example arguments:  /home/pnb/trainingOutput/pytorchOutput_A10/ /home/pnb/trainingData/A10/test_images/ /home/pnb/trainingOutput/pytorchOutput_A10/infer_test_images/
# source inference_many_models.sh /home/pnb/trainingOutput/pytorchOutputFtoM_A10/ /home/pnb/trainingData/A10/test_masks/ /home/pnb/trainingOutput/pytorchOutputFtoM_A10/infer_mask_images/
# source inference_many_models.sh /home/pnb/trainingOutput/pytorchOutputFtoM_A10/ /home/pnb/trainingData/A10/train_masks/ /home/pnb/trainingOutput/pytorchOutputFtoM_A10/infer_mask_images/

model_directory=$1
image_directory=$2
output_directory=$3

#echo 'name_dataset: '$name_dataset
echo 'model_directory: '$model_directory
echo 'image_directory: '$image_directory
echo 'output_directory: '$output_directory

for file in $(find $model_directory); do
  #echo 'file :'$file
  if [ -d $file ]; then
    # do something directory-ish
    echo 'directory: '$file
  else
    if [[ $file == *.pt ]]; then
      # process the model .pt file
      file_basename="${file##*/}"
      echo $file_basename
      # remove the .pt suffix and create a folder for the model
      len=$(expr length "$file_basename")
      #echo "The length of string is $len"
      model_specific_output=${file_basename:0:len-3}
      output_dir=${output_directory}"/"${model_specific_output}
      #echo 'inside outdir :'$output_dir
      if [ ! -d $output_dir ]; then
        #echo 'create a folder :'$output_dir
        mkdir -p $output_dir
      fi

      #echo 'input parameters:'
      echo '--model_filepath'${file} '--image_dirpath'${image_directory} '--output_dirpath'${output_dir}
      if [[ ${file_basename} != *"unet"* ]]; then
        echo 'runnimg pytorch model inference'
        python pytorch_models/inference_INFER.py --model_filepath=${file} --image_dirpath=${image_directory} --output_dirpath=${output_dir}
      else
        echo 'running unet model inference'
        python UNet/infer_dataset.py --model_filepath=${file} --image_dirpath=${image_directory} --output_dirpath=${output_dir}

      fi
    fi
  fi
done


