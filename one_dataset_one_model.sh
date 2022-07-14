#!/bin/bash

learning_rate=$1
model_filename=$2
mn=$3
model_name=$4
pretrained=$5
name_dataset=$6
num_classes=$7
echo "number of classes: "${num_classes}

data="/home/pnb/trainingData/"${name_dataset}
train_images='train_images'
train_masks='train_masks'
test_images="test_images"
test_masks="test_masks"
#train_images='train_masks'
#train_masks='train_masks'
#test_images="test_masks"
#test_masks="test_masks"
output_dir="/home/pnb/trainingOutput/pytorchOutputMtoM_"${name_dataset}
device_name="gpu"
epochs=100
batch_size=2

echo "number of epochs: "${epochs}

echo "About to start running ${mn}"
if [[ ${model_name} != "unet" ]]
then
 echo 'pytorch model'
 echo 'input parameters:'
 echo '--data='${data}' --train_images='${train_images}' --train_masks='${train_masks}' --test_images='${test_images}' --test_masks='${test_masks}
 echo '--output_dir='${output_dir}' --epochs='${epochs}' --model_filename='${model_filename}' --device_name='${device_name}' --batch_size='${batch_size}
 echo '--learning_rate='${learning_rate}' --metrics_name='${mn}' --model_name='${model_name}' --pretrained='${pretrained}' --classes='${num_classes}
 python pytorch_models/train.py --data=${data} --train_images=${train_images} --train_masks=${train_masks} --test_images=${test_images} --test_masks=${test_masks} --output_dir=${output_dir} --epochs=${epochs} --model_filename=${model_filename} --device_name=${device_name} --batch_size=${batch_size} --learning_rate=${learning_rate} --metrics_name=${mn} --model_name=${model_name} --pretrained=${pretrained} --classes=${num_classes}
else
  echo 'unet model'
  t_train_image_folder=${data}"/"${train_images}
  t_train_mask_folder=${data}"/"${train_masks}
  t_test_image_folder=${data}"/"${test_images}
  t_test_mask_folder=${data}"/"${test_masks}
  lmdb_output_folder=${data}"/LmdbCreation"
  #echo ${t_train_image_folder}
  #echo ${lmdb_output_folder}

  echo 'input parameters for lmdb:'
  echo '--image_folder='${t_train_image_folder}' --test_image_folder='${t_test_image_folder}' --mask_folder='${t_train_mask_folder}
  echo '--test_mask_folder='${t_test_mask_folder}' --output_folder='${lmdb_output_folder}

  python UNet/build_lmdb.py --image_folder=${t_train_image_folder} --test_image_folder=${t_test_image_folder} --mask_folder=${t_train_mask_folder} --test_mask_folder=${t_test_mask_folder} --output_folder=${lmdb_output_folder}

  output_directory=${output_dir}
  train_database=${lmdb_output_folder}"/train-HES.lmdb"
  test_database=${lmdb_output_folder}"/test-HES.lmdb"

  number_of_epochs=${epochs}

  echo 'input parameters for unet training:'
  echo '--batch_size='${batch_size}' --learning_rate='${learning_rate}' --output_dir='${output_directory}' --train_database='${train_database}
  echo '--test_database='${test_database}' --metrics_name='${mn}' --model_filename='${model_filename}' --cfname='${cfname}
  echo '--number_classes='${num_classes}' --number_of_epochs='${number_of_epochs}

  python UNet/train_lmdb_dataset.py --batch_size=${batch_size} --learning_rate=${learning_rate} --output_dir=${output_directory} --train_database=${train_database} --test_database=${test_database} --metrics_name=${mn} --model_filename=${model_filename} --cfname=${cfname} --number_classes=${num_classes} --number_of_epochs=${number_of_epochs}

fi
