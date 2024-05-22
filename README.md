# Software: Characterization of AI Model Configurations for Model Reuse

## Statements of purpose and maturity

The purpose of this work is to increase reusability of trained AI models via establishing
and “standardizing” metrics that

- (a) would be useful to a third party reusing trained AI models and
- (b) would still protect trade secrets of the party providing trained AI models.

The optimization curves gathered during AI model training provide such a source of information
and the data points in optimization curves could meet both objectives (usefulness and protection).

The motivation of this project is to implement baseline AI model metrics from train and test
optimization curves gathered during training. These metrics would be included with disseminated
trained AI models in AI Model Cards and would support better reuse of shared trained AI models.

In addition, these metrics can be used for ranking AI architectures in terms of their suitability
to specific scientific applications
(i.e., a recommendation system for AI architectures suitable to scientific applications).
The metrics are focused on model (accuracy, stability),
training process (speed, predictability, initialization gain),
training data (uniformity of training data, pretrained data compatibility with domain data, pretrained data
compatibility with model architecture), and
and hardware (GPU RAM memory usage, energy consumption).

## Description of the repository contents

- pytorch_models: contains the training and inference code for multiple image segmentation AI architectures
  supported by PyTorch library
  **pytorch_models/train.py** - training and evaluation loop for a train and test dataset. Calculates many metrics of
  accuracy
  that will be saved into a csv file that will be located in the specified output directory. See parameter descriptions
  for details.
  Note: batch size of 1 does not work with pytorch models. Batch size must be >= 2

  **pytorch_models/datahandler.py** - this file eases the dataset creation process. It basically creates a train and
  test dataset
  and puts them into a "dataloader" dictionary for easy access. This file requires no arguments/editing. It runs
  in the background.

  **pytorch_models/segdataset.py** - this is the file that creates the pytorch datasets. It contains the init, getitem,
  and len
  functions, which are required for pytorch dataloaders. Just like datahandler, this file runs in the background
  and requires no direct attention

  **pytorch_models/inference.py** - this file runs inference on a set of test images (or train images if desired). Given
  a model
  weights file, inference.py will run inference on all images in a folder, and save them in an output directory.

- UNet: contains the UNet model training code

  **UNet/train_lmdb_dataset.py** - This is the training file for UNet. It takes in an lmdb dataset containing
  a train and test set, and runs training/validation on the dataset.

  **UNet/build_lmdb.py** - This is the file that builds the lmdb dataset required to train models. Check
  parameters for more detals.

  **UNet/unet_model.py** - This is the UNet model implementation.

  **UNet/unet_dataset.py** - This is the file that creates a pytorch dataset that eventually gets passed
  into the dataloader. It contains the init, getitem, and len functions.

  **UNet/(isg_ai_pb2.py)(isg_ai_proto.txt)** - These files are for the google protobuf compiler, no need
  to pay attention to these files.

  **UNet/augment.py** - This file is not used for training. It was pulled from Michael Majurski's original code.

- preprocess: contains code for tiling and stitching images, evaluating signal-to-noise ratios of images, renaming
  files,
  manipulating mask labels, fusing labels with background, and inpainting regions

  **tiling.py** - this tool tiles images without overlap. For example, an image named "example" can be tiled into 2x2
  pieces, resulting in 4 images named example-0_0, example-0_1,
  example-1_0, and example_1_1. This tool is useful when a batch size of 2 or more does not fit into GPU RAM

  **split.py** - this tool randomly splits an image collection into train/test subsets based on a fraction
  (a value between 0 and 1). For example, a fraction of .8 will have 80% of the
  dataset fall into the train dataset, and 20% into the test dataset. See parameter descriptions for details

- graph_and_fit: contains code for creating visualization of metrics derived from optimization curves
  from multiple trained AI models. Within the folder, there is a power_curves.py file that automatically
  generates graphs from a folder of csv files. Please read the documentation in the README within that
  folder.

- root directory: contains shell scripts for launching training and inference on one or many models with
  one or many training datasets

  **one_dataset_one_model.sh** - This bash script is for training one model with one dataset.
  `  
  source one_dataset_one_model.sh ${learning_rate} "lraspp_model_${name_dataset}_${counter}.pt" "lraspp_metrics_${name_dataset}_${counter}.csv" "LR-ASPP-MobileNetV3-Large" "False" "${name_dataset}" ${num_classes}
  `

  **one_dataset_many_models.sh** - This bash script is for training many models with one training dataset.
  It does not take any arguments. However, the values for two variables must be set in the bash script
  `
  name_dataset="infer3_sep9_contrast"
  num_classes=9
  `

  **inference_many_models.sh** - This bash script is for inferencing
  `
  source inference_many_models.sh <folder with models, e.g., ./pytorchOutputFtoM_A10/>
  <folder with input test images, e.g., ./trainingData/A10/test_masks/>
  <folder with output mask images, e.g., ./infer_mask_images/>
  `

  **inference_tiles_many_models.sh** - This bash script is for limited GPU RAM. THe input files
  are inferenced like using inference_many_models.sh and then they are stitch together to the
  original size.
  `
  source inference_tiles_many_models.sh <folder with models, e.g., ./pytorchOutputFtoM_A10/>
  <folder with input test images, e.g., ./trainingData/A10/test_masks/>
  <folder with output mask images, e.g., ./infer_mask_images/>
  `

### Technical installation instructions, including operating system or software dependencies

The project is leveraging [Torchvision models](https://pytorch.org/vision/stable/models.html) libraries containing
implementations of multiple artificial intelligence (AI) models. It has been developed on Linux OS, Ubuntu 18.04

#### Installation

- use requirements.txt for pip installation or requirements_conda.txt for conda installation
    - pip install -r requirements.txt
    - conda create --name <env> --file requirements_conda.txt
- if the installation fails, then you can use a set of pip installs:
    - conda create -n airec_test python=3.8 (or conda create --prefix <dir><nn-util> python=3.8)
    - conda activate airec_test (or conda activate <dir><airec_test>)
    - conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
    - (try -c conda-forge if that doesnt work)
    - sometimes needed: pip install numpy==1.19.2 Cython==3.0a1
    - pip install scikit-image
    - pip install GPUtil (or conda install -c conda-forge gputil )
    - pip3 install -U scikit-learn
    - pip install tqdm
    - pip install lmdb (needed by Unet)
    - conda install protobuf (needed by UNet)
    - pip install pandas (needed for plotting and comparisons)
    - pip install matplotlib (needed for plotting)
    - pip install plotly (needed for plotting)
    - pip install -U kaleido (needed for plotly)

#### Execution

- step 1: obtain optimization curves

`
run one of the scripts in the root directory
`

- step 2: extract metrics from optimization curves

`
run the script predict_compare.sh in the graph_and_fit directory
`

- step 3: visualize multiple metrics from multiple trained AI models to support decision making and efficient AI Mode
  reuse

`
exlore the graphs in the <target/graphs> directory and the comparisons
in the <target/comparisons> directory
`

#### INFER data

For INFER data follow the same steps above but use files with 'INFER' affix, wherever it exists. For example, use
INFER_stitching.py instead of stitching.py.

Mask data is 2D and hence can follow the same sequence of steps as that for other 2D data. In the preprocessing step,
combine_and_tile.py should replace the use of tiling.py for sequence of image data such as INFER.

# Contact information

- Peter Bajcsy, ITL NIST, Software and System Division, Information Systems Group
- Contact email address at NIST: peter.bajcsy@nist.gov

# Credits:

- The contributions to the code in this repository came from:
    - *Peter Bajcsy*
    - *Pushkar Sathe*
    - *Daniel Gao*
    - *Ivy Liang*
    - *Michael Majurski*

# Related Material

- URL for associated project on the NIST website: https://www.nist.gov/itl/ssd/information-systems-group
- URL for Model Cards toolkit: https://github.com/tensorflow/model-card-toolkit

[comment]: # ( References to user guides if stored outside of GitHub)

# Citation:

- Peter Bajcsy, Michael Majurski, Thomas E. Cleveland IV, Manuel Carrasco, Walid Keyrouz,
  “Characterization of AI Model Configurations for Model Reuse,”
  Bio Image Computing workshop, European Conference on Computer Vision (ECCV), 2022,
  24-28 October 2022 Tel-Aviv, Israel.
#####
- Pushkar S. Sathe, Caitlyn M. Wolf, Youngju Kim, Sarah M. Robinson, M. Cyrus Daugherty, Ryan P. Murphy, Jacob M.
  LaManna, Michael G. Huber, David L. Jacobson, Paul A. Kienzle, Katie M. Weigandt, Nikolai N. Klimov, Daniel S.
  Hussey & Peter Bajcsy
  "Data-driven simulations for training AI-based segmentation of neutron images." Scientific Reports, 14(1), 6614.

[comment]: # ( References to any included non-public domain software modules, and additional license language if needed, e.g. BSD, GPL, or MIT)





