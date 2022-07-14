# Code for preprocessing data before running AI model training sessions 

### Tiling to meet the batch size limits due to GPU RAM size
 
python tiling.py --image_dir /home/pnb/trainingData/A10/images --output_dir
/home/pnb/trainingData/A10/tiled_images --xPieces 2 --yPieces 2

Note: The arguments refer to the number of cuts along x- and y-axes (xPieces and yPieces). 
Thus, an input image of size 1024 x 1024 will be cut into tiles of size 512 x 512 with arguments
--xPieces 2 --yPieces 2

### Split images and masks into train and test subsets
python split.py --image_dir /home/pnb/trainingData/A10/tiled_images --mask_dir /home/pnb/trainingData/A10/tiled_masks
--train_image_dir=/home/pnb/trainingData/A10/train_images --train_mask_dir=/home/pnb/trainingData/A10/train_masks
--test_image_dir=/home/pnb/trainingData/A10/test_image --test_mask_dir=/home/pnb/trainingData/A10/test_masks --fraction 0.8

Note: the bash script tile_and_split.sh will run both tiling and split.

### Utility 1: Generate a histogram plot for a grayscale image/mask and save histogram plots and porosity summary CSV file 
python class_histogram.py --image_dir --max_labels --output_dir --save_hist 

### Utility 2: Convert masks in .npy file format to .png file format  
python mask_manipulation.py --image_dir --output_dir  

