# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.
import csv
import os
import argparse
import time

import GPUtil
import sys
from numpy import unicode
from datetime import datetime

"""
This class will collect information about GPU utilization

"""


def write_header(output_filename):
    fieldnames = ['Epoch', 'Time stamp', 'ID', 'Name', 'Serial', 'UUID', 'GPU temp. [C]', 'GPU util. [%]',
                  'Memory util. [%]',
                  'Memory total [MB]', 'Memory used [MB]', 'Memory free [MB]', 'Display mode', 'Display active']
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


def record(epoch, output_filename):
    # out_name, out_file_extension = os.path.splitext(output_filename)
    output_dir = os.path.dirname(output_filename)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    GPUs = GPUtil.getGPUs()
    print('INFO: GPUs:', GPUs)
    if len(GPUs) < 1:
        print('WARNING: the hardware does not contain NVIDIA GPU card')
        return

    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y:%m:%d:%H:%M:%S")
    print('INFO: date_time ', date_time)

    attrList = [[{'attr': 'id', 'name': 'ID'},
                 {'attr': 'name', 'name': 'Name'},
                 {'attr': 'serial', 'name': 'Serial'},
                 {'attr': 'uuid', 'name': 'UUID'}],
                [{'attr': 'temperature', 'name': 'GPU temp.', 'suffix': 'C', 'transform': lambda x: x, 'precision': 0},
                 {'attr': 'load', 'name': 'GPU util.', 'suffix': '%', 'transform': lambda x: x * 100, 'precision': 0},
                 {'attr': 'memoryUtil', 'name': 'Memory util.', 'suffix': '%', 'transform': lambda x: x * 100,
                  'precision': 0}],
                [{'attr': 'memoryTotal', 'name': 'Memory total', 'suffix': 'MB', 'precision': 0},
                 {'attr': 'memoryUsed', 'name': 'Memory used', 'suffix': 'MB', 'precision': 0},
                 {'attr': 'memoryFree', 'name': 'Memory free', 'suffix': 'MB', 'precision': 0}],
                [{'attr': 'display_mode', 'name': 'Display mode'},
                 {'attr': 'display_active', 'name': 'Display active'}]]

    # store the date_time as teh first entry in the recorded row
    store_gpu_info = str(epoch) + ',' + date_time

    for attrGroup in attrList:
        # print('INFO: attrGroup:', attrGroup)

        index = 1
        for attrDict in attrGroup:
            attrPrecision = '.' + str(attrDict['precision']) if ('precision' in attrDict.keys()) else ''
            attrTransform = attrDict['transform'] if ('transform' in attrDict.keys()) else lambda x: x

            for gpu in GPUs:
                attr = getattr(gpu, attrDict['attr'])

                attr = attrTransform(attr)

                if (isinstance(attr, float)):
                    attrStr = ('{0:' + attrPrecision + 'f}').format(attr)
                elif (isinstance(attr, int)):
                    attrStr = ('{0:d}').format(attr)
                elif (isinstance(attr, str)):
                    attrStr = attr;
                elif (sys.version_info[0] == 2):
                    if (isinstance(attr, unicode)):
                        attrStr = attr.encode('ascii', 'ignore')
                else:
                    raise TypeError(
                        'Unhandled object type (' + str(type(attr)) + ') for attribute \'' + attrDict['name'] + '\'')

                # print('INFO: attrStr ', attrStr)
                store_gpu_info += ',' + attrStr
                index += 1

    store_gpu_info += '\n'
    print('row data:', store_gpu_info)
    with open(output_filename, 'a', newline='') as csvfile:
        csvfile.write(store_gpu_info)


def main():
    parser = argparse.ArgumentParser(prog='record gpu utilization',
                                     description='Script that collects information about GPU utilization')
    parser.add_argument('--output_filename', type=str, help='filename for saving output statistics')
    args, unknown = parser.parse_known_args()

    if args.output_filename is None:
        print('ERROR: missing output_filename ')
        return

    write_header(args.output_filename)
    record(1, args.output_filename)
    time.sleep(1)
    record(2, args.output_filename)
    time.sleep(2)
    record(3, args.output_filename)
    time.sleep(3)
    record(4, args.output_filename)


if __name__ == "__main__":
    main()
