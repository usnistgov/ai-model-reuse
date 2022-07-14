# Code for processing all metrics gathered from AI model training sessions 

- Necessary Packages
The user must have pandas, numpy, matplotlib, and scipy packages already installed on their device.

### Step 1: Generate prediction values base don power and exponential models and n-points 
python power_function_fit.py --input_dir /home/pnb/trainingOutput/pytorchOutput_A10
--output_dir /home/pnb/trainingOutput/pytorchOutput_A10/graphs

The output will be:
- a set of graphs in png file format
- a set of CSV files with the original and predicted values for each combination of model and n-points
- a CSV file named 'coefficients.csv' with the following entries:

CSV File Name	Point Selection	power multiplier	power exponent	exp multiplier	exp base

### Step 2: Generate one row per model so that models can be rank-ordered
python comparison_metrics.py --input_dir /home/pnb/trainingOutput/pytorchOutput_A10/graphs
--output_dir /home/pnb/trainingOutput/pytorchOutput_A10/comparisons

### Step 3: Generate statistics (one row per model) for GPU utilization so that models can be rank-ordered
python gpu_metrics.py --input_dir /home/pnb/trainingOutput/pytorchOutput_concrete/gpu_info/ 
--output_dir /home/pnb/trainingOutput/pytorchOutput_concrete/comparisons/

### Note:
predict_compare.sh executes all three steps in a batch mode

### Utility 1: Generate a histogram plot for a grayscale image/mask and save histogram plots and porosity summary CSV file 
python class_histogram.py --image_dir --max_labels --output_dir --save_hist 

### Utility 2: Convert masks in .npy file format to .png file format  
python mask_manipulation.py --image_dir --output_dir  


