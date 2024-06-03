# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

from torch.utils.data import DataLoader
from torchvision import transforms

import segdataset


class GetDataloader():
    def __init__(self, data_dir, train_image_folder, train_mask_folder, test_image_folder, test_mask_folder, fraction,
                 batch_size, n_classes):
        self.data_dir = data_dir
        self.train_image_folder = train_image_folder
        self.train_mask_folder = train_mask_folder
        self.test_image_folder = test_image_folder
        self.test_mask_folder = test_mask_folder
        self.fraction = fraction
        self.batch_size = batch_size
        self.n_classes = n_classes
        # data_transforms = transforms.Compose([transforms.ToTensor(),
        #                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        data_transforms = transforms.Compose([transforms.ToTensor()])
        image_datasets = {
            x: segdataset.SegmentationDataset(data_dir,
                                              train_image_folder=train_image_folder,
                                              train_mask_folder=train_mask_folder,
                                              test_image_folder=test_image_folder,
                                              test_mask_folder=test_mask_folder,
                                              fraction=fraction,
                                              subset=x,
                                              transforms=data_transforms,
                                              n_classes=n_classes)
            for x in ['Train', 'Test']
        }
        self.weights = image_datasets['Train'].weights
        dataloaders = {
            x: DataLoader(image_datasets[x],
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=8, drop_last=True)
            for x in ['Train', 'Test']
        }
        self.dataloaders = dataloaders
