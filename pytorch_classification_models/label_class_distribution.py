# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.
import os
import argparse
import pandas as pd

"""
This class will compute the histogram of the number of example images per image class 
from a set of folders with subfolder names indicating the class label

This was developed for the Kaggle PANDA biopsy images 

Script that surveys a folder with subfolders containing varying number of class specific sub-folders
__author__      = "Peter Bajcsy"
__email__ = "peter.bajcsy@nist.gov"
"""

def survey(dir_name, number_labels):
    # prepare the counters
    count_samples = []
    for i in range(0,number_labels):
        count_samples.append(int(0))

    if os.path.isdir(dir_name[0]):
        for i in range(0,number_labels):
            # dir_name[1] contains all subdirectories
            for j in range(0, len(dir_name[1])):
                if dir_name[1][j].endswith(str(i)):
                    DIR = os.path.join(dir_name[0],dir_name[1][j])
                    count_samples[i] = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
                    print('INFO: dir_name=', dir_name[1][j], ' count for class label i=', i, ' is ',  count_samples[i] )

    return count_samples

def survey_batch(input_dir, number_labels, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    out_stats = 'dir_class_stats.csv'
    out_dir_class_stats = os.path.join(output_dir, out_stats)
    directory_name = 'directory name'
    header_names = []
    header_names.append(directory_name)
    for i in range(0, number_labels):
        header_names.append('class label ' + str(i))

    metrics_stats = pd.DataFrame(columns=header_names)
    header_col = metrics_stats.columns
    print('DEBUG: header_col=', header_col)

    # create the output file only if it does not exist
    if not os.path.exists(out_dir_class_stats):
        metrics_stats.to_csv(out_dir_class_stats, mode='w', header=True, index=False)

    for parent_dirname in os.walk(input_dir):
        if os.path.isdir(parent_dirname[0]):
            if not parent_dirname[0].endswith('_mask'):
                # ignore the mask folders
                continue

            #dirname_array.append(parent_dirname[0])

            #for dir_name in os.walk(parent_dirname[0]):
            count_samples = survey(parent_dirname, number_labels)

            print('DEBUG: count_samples=', count_samples)

            metrics_stats[header_col[0]] = [parent_dirname[0]]
            for i in range(0,number_labels):
                metrics_stats[header_col[i+1]] = [count_samples[i]]

            metrics_stats.to_csv(out_dir_class_stats, mode='a', header=False, index=False)

def main():
    parser = argparse.ArgumentParser(prog='label class hist', description='Script that surveys a folder with subfolders containing vaying number of class specific sub-folders')
    parser.add_argument('--input_dir', type=str, help='folder path to input folder')
    parser.add_argument('--output_dir', type=str, help='folder path to saving output tiles')
    args, unknown = parser.parse_known_args()

    if args.input_dir is None:
        print('ERROR: missing input image dir ')
        return

    if args.output_dir is None:
        print('ERROR: missing output dir ')
        return

    # this is for the Kaggle H&E stained images with Gleason score in [0,5]
    number_labels = 6
    survey_batch(args.input_dir, number_labels, args.output_dir)

if __name__ == "__main__":
    main()