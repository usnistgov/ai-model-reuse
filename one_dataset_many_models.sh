#!/bin/bash

counter=1
while [ $counter -le 4 ]
do
  if [ $counter -eq 1 ]
  then
    learning_rate=1e-5
  fi

  if [ $counter -eq 2 ]
  then
    learning_rate=1e-4
  fi

  if [ $counter -eq 3 ]
  then
    learning_rate=1e-3
  fi

  if [ $counter -eq 4 ]
  then
    learning_rate=1e-2
  fi

  # infer14 - 9, concrete - 4, rpe2d, other - 2
  name_dataset="infer3_sep9_contrast"
  num_classes=9
  source one_dataset_one_model.sh ${learning_rate} "deeplab50_model_${name_dataset}_${counter}.pt" "deeplab50_metrics_${name_dataset}_${counter}.csv" "Deeplab50" "False" "${name_dataset}" ${num_classes}
  source one_dataset_one_model.sh ${learning_rate} "deeplab50_model_${name_dataset}_${counter}_pretrained.pt" "deeplab50_metrics_${name_dataset}_${counter}_pretrained.csv" "Deeplab50" "True" ${name_dataset} ${num_classes}
  source one_dataset_one_model.sh ${learning_rate} "deeplab101_model_${name_dataset}_${counter}.pt" "deeplab101_metrics_${name_dataset}_${counter}.csv" "Deeplab101" "False" "${name_dataset}" ${num_classes}
  source one_dataset_one_model.sh ${learning_rate} "deeplab101_model_${name_dataset}_${counter}_pretrained.pt" "deeplab101_metrics_${name_dataset}_${counter}_pretrained.csv" "Deeplab101" "True" ${name_dataset} ${num_classes}
  source one_dataset_one_model.sh ${learning_rate} "fcn_resnet50_model_${name_dataset}_${counter}.pt" "fcn_resnet50_metrics_${name_dataset}_${counter}.csv" "Resnet50" "False" "${name_dataset}" ${num_classes}
  source one_dataset_one_model.sh ${learning_rate} "fcn_resnet50_model_${name_dataset}_${counter}_pretrained.pt" "fcn_resnet50_metrics_${name_dataset}_${counter}_pretrained.csv" "Resnet50" "True" ${name_dataset} ${num_classes}
  source one_dataset_one_model.sh ${learning_rate} "fcn_resnet101_model_${name_dataset}_${counter}.pt" "fcn_resnet101_metrics_${name_dataset}_${counter}.csv" "Resnet101" "False" "${name_dataset}" ${num_classes}
  source one_dataset_one_model.sh ${learning_rate} "fcn_resnet101_model_${name_dataset}_${counter}_pretrained.pt" "fcn_resnet101_metrics_${name_dataset}_${counter}_pretrained.csv" "Resnet101" "True" ${name_dataset} ${num_classes}
  source one_dataset_one_model.sh ${learning_rate} "mobilenetv3_model_${name_dataset}_${counter}.pt" "mobilenetv3_metrics_${name_dataset}_${counter}.csv" "MobileNetV3-Large" "False" "${name_dataset}" ${num_classes}
  source one_dataset_one_model.sh ${learning_rate} "lraspp_model_${name_dataset}_${counter}.pt" "lraspp_metrics_${name_dataset}_${counter}.csv" "LR-ASPP-MobileNetV3-Large" "False" "${name_dataset}" ${num_classes}

  source one_dataset_one_model.sh ${learning_rate} "unet_model_${name_dataset}_${counter}.pt" "unet_metrics_${name_dataset}_${counter}.csv" "unet" "False" "${name_dataset}" ${num_classes}

##  # TODO debug the initialization of these models
#  source one_dataset_one_model.sh ${learning_rate} "mobilenetv3_model_${name_dataset}_${counter}_pretrained.pt" "mobilenetv3__metrics_${name_dataset}_${counter}_pretrained.csv" "MobileNetV3-Large" "True" ${name_dataset} ${num_classes}
#  source one_dataset_one_model.sh ${learning_rate} "lraspp_model_${name_dataset}_${counter}_pretrained.pt" "lraspp_metrics_${name_dataset}_${counter}_pretrained.csv" "LR-ASPP-MobileNetV3-Large" "True" ${name_dataset} ${num_classes}
  ((counter++))
done