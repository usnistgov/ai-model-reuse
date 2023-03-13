import torch

from torchsummary import summary  #: For asserting model input output layer dimensions


class INFERFeatureExtractor(torch.nn.Module):
    """
    This model is made for the purpose of using 3 dimensional data as inputs
    where the last 2 dimensions are X and Y axes of images.
    TODO: add support for upto 5 dimensions.

    concatenating with other publicly available models.
    This model is not intented to be used on its own

    """

    def __init__(self, input_channels):
        super(INFERFeatureExtractor, self).__init__()
        self.input_channels = input_channels
        # self.height = shape[-2]
        # self.width = shape[-1]

        self.layer1 = torch.nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.activation1 = torch.nn.ReLU(inplace=True)
        # self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = torch.nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        # self.layer2 = torch.nn.Conv2d(in_channels=6, out_channels=1, kernel_size=(6, 5), stride=1,
        #                               padding='same')

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        # x = self.pool1(x)
        x = self.layer2(x)
        return x


class CombinedModel(torch.nn.Module):
    """
    This model is made for the purpose of using 3 dimensional data as inputs
    where the last 2 dimensions are X and Y axes of images.
    TODO: add support for upto 5 dimensions.

    concatenating with other publicly available models.
    This model is not intented to be used on its own

    """

    def __init__(self, input_shape, segmentation_model, n_classes):
        super(CombinedModel, self).__init__()
        self.models = []
        self.feature_extractor = INFERFeatureExtractor(input_shape)
        # self.segmentation_model = models.segmentation.deeplabv3_resnet50(num_classes=n_classes, pretrained=True)
        self.segmentation_model = segmentation_model
        # self.concat_layer = torch.nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        # self.final_layer = torch.nn.Conv2d(256, n_classes, kernel_size=1, stride=1, padding=0)

    # def combine(self, *listofmodels):
    #     self.models.append(listofmodels)  # TEST

    def forward(self, x):
        features = self.feature_extractor(x)
        segmentation_output = self.segmentation_model(features)
        segmentation_output = segmentation_output['out']
        # concat_output = torch.cat((features, segmentation_output), dim=1)
        # concat_output = self.concat_layer(concat_output)
        # final_output = self.final_layer(concat_output)
        return segmentation_output
        # for selected_model in self.models:
        #     x = selected_model(x)
        # return x


if __name__ == "__main__":
    """
    Example architecture: INFER_Model_cn1 + Deeplab50
    
    To incorporate autocorrelation and other parameters the following options seemto be  worth looking into:
        1. ensembling/ combining the segmentation output with a small network that accepts the output 
        from combined net image parameters
        2. multi input for first block : https://rosenfelder.ai/multi-input-neural-network-pytorch/
    """
    from torchvision import models

    outputchannels = 2
    input_channels = 252
    # image_shape = (252, 2000, 2000)
    infer_input_model = INFERFeatureExtractor(252)
    preset_model = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=outputchannels, progress=True)
    combined_model = CombinedModel(input_shape=input_channels, segmentation_model=preset_model,
                                   n_classes=outputchannels)
    print(summary(infer_input_model, (252, 96, 96)))
    # print(summary(combined_model, (252, 96, 96)))
    # exit()
    import numpy as np

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(combined_model.parameters(), lr=0.001)
    num_epochs = 1
    combined_model.train()
    for epoch in range(num_epochs):
        input_data = torch.from_numpy(np.random.random((2, 252, 96, 96)).astype(np.float32))
        target = torch.from_numpy((np.random.random((2, outputchannels, 96, 96)) > 0.3).astype(np.float32))
        outputs = combined_model(input_data)
        loss = criterion(outputs, target)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # encoded = infer_input_model(input)
    # output = modelB(x2)
    # augmented_preset_model = CombinedModel().combine([infer_input_model, preset_model])
    # augmented_preset_model = torch.cat([infer_input_model, preset_model], dim=1)
    # preset_model.detach() if weights not to be changed
    """
    Necessary to fix conv1d not working
    !pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
    """
