# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch
import torch.nn as nn


def double_conv(fc_in,fc_out,kernel,stride=1):
    stage = torch.nn.Sequential()
    pad = (kernel - 1) // 2
    stage.add_module('conv', torch.nn.Conv2d(in_channels=fc_in,
                                             out_channels=fc_out, kernel_size=kernel, stride=stride,
                                             padding=pad, bias=False))
    stage.add_module('batch_norm', torch.nn.BatchNorm2d(fc_out))
    stage.add_module('relu', torch.nn.ReLU(inplace=True))
    stage.add_module('conv', torch.nn.Conv2d(in_channels=fc_in,
                                             out_channels=fc_out, kernel_size=kernel, stride=stride,
                                             padding=pad, bias=False))
    stage.add_module('batch_norm', torch.nn.BatchNorm2d(fc_out))
    stage.add_module('relu', torch.nn.ReLU(inplace=True))
    stage.add_module('pool',torch.nn.MaxPool2d(2))
    return stage

def double_deconv(fc_in,fc_out,kernel,stride=1):
    stage = torch.nn.Sequential()
    pad = (kernel - 1) // 2
    stage.add_module('upsample', torch.nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True))
    stage.add_module('deconv', torch.nn.ConvTranspose2d(in_channels=fc_in,
                                             out_channels=fc_out, kernel_size=kernel, stride=stride,
                                             padding=pad, bias=False))
    stage.add_module('batch_norm', torch.nn.BatchNorm2d(fc_out))
    stage.add_module('relu', torch.nn.ReLU(inplace=True))
    stage.add_module('deconv', torch.nn.ConvTranspose2d(in_channels=fc_in,
                                             out_channels=fc_out, kernel_size=kernel, stride=stride,
                                             padding=pad, bias=False))
    stage.add_module('batch_norm', torch.nn.BatchNorm2d(fc_out))
    stage.add_module('relu', torch.nn.ReLU(inplace=True))
    return stage

def bottleneck(fc_in,fc_out,kernel,stride=1):
    stage = torch.nn.Sequential()
    pad = (kernel - 1) // 2
    stage.add_module('conv', torch.nn.Conv2d(in_channels=fc_in,
                                             out_channels=fc_out, kernel_size=kernel, stride=stride,
                                             padding=pad, bias=False))
    stage.add_module('batch_norm', torch.nn.BatchNorm2d(fc_out))
    stage.add_module('relu', torch.nn.ReLU(inplace=True))
    stage.add_module('conv', torch.nn.Conv2d(in_channels=fc_in,
                                             out_channels=fc_out, kernel_size=kernel, stride=stride,
                                             padding=pad, bias=False))
    stage.add_module('batch_norm', torch.nn.BatchNorm2d(fc_out))
    stage.add_module('relu', torch.nn.ReLU(inplace=True))
    stage.add_module('dropout',torch.nn.Dropout(0.5))
    return stage

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.mdict = self.build_net_dict(self.n_channels)

    def build_net_dict(self,n_channels):
        mdict = torch.nn.ModuleDict()
        out_channels = 64
        mdict['down_1'] = double_conv(n_channels, out_channels, 3, stride=1)
        mdict['down_2'] = double_conv(out_channels, 2*out_channels, 3, stride=1)
        mdict['down_3'] = double_conv(2*out_channels, 4*out_channels, 3, stride=1)
        mdict['down_4'] = double_conv(4*out_channels, 8*out_channels, 3, stride=1)
        mdict['down_5'] = bottleneck(8*out_channels,16*out_channels, 3, stride=1)
        mdict['up_4'] = double_deconv(16*out_channels, 8*out_channels, 3, stride=1)
        mdict['up_3'] = double_deconv(8*out_channels, 4*out_channels, 3, stride=1)
        mdict['up_2'] = double_deconv(4*out_channels, 2*out_channels, 3, stride=1)
        mdict['up_1'] = double_deconv(2*out_channels, out_channels, 3, stride=1)
        mdict['outconv'] = torch.nn.Conv2d(out_channels, self.n_classes, 1)
        return mdict

    def forward(self, x):
        x = self.mdict['down_1'](x)
        x = self.mdict['down_2'](x)
        x = self.mdict['down_3'](x)
        x = self.mdict['down_4'](x)
        x = self.mdict['down_5'](x)
        x = self.mdict['up_4'](x)
        x = self.mdict['up_3'](x)
        x = self.mdict['up_2'](x)
        x = self.mdict['up_1'](x)
        x = self.mdict['outconv'](x)
        return x
