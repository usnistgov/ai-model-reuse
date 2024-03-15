import torch

from torchsummaryX import summary as summaryX  #: For asserting model input output layer dimensions
from torchsummary import summary  #: For asserting model input output layer dimensions


class INFERFeatureExtractor(torch.nn.Module):
    """
    This model is made for the purpose of using 3 dimensional data as inputs
    where the last 2 dimensions are X and Y axes of images.

    It does so by concatenating with other publicly available models.
    This model is not intended to be used on its own

    """

    def __init__(self, input_channels):
        super(INFERFeatureExtractor, self).__init__()
        self.input_channels = input_channels
        self.output_channels = 3

        self.layer1 = torch.nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=1, padding=1)
        # self.bn1 = torch.nn.BatchNorm2d(32)
        self.activation1 = torch.nn.ReLU(inplace=True)
        # self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = torch.nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        self.activation2 = torch.nn.ReLU(inplace=True)
        # self.layer2 = torch.nn.Conv2d(in_channels=6, out_channels=1, kernel_size=(6, 5), stride=1,
        #                               padding='same')

    def forward(self, x):
        x = self.layer1(x)
        # x = self.bn1(x)
        x = self.activation1(x)
        # x = self.pool1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        return x


class INFERFeatureExtractor1D(torch.nn.Module):
    """
    This model is made for the purpose of using 3 dimensional data as inputs
    where the last 2 dimensions are X and Y axes of images.
    TODO: add support for upto 5 dimensions.

    concatenating with other publicly available models.
    This model is not intented to be used on its own

    """

    def __init__(self, input_channels, standalone=True, output_channels=3):
        super(INFERFeatureExtractor1D, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.layer1 = torch.nn.Conv1d(self.input_channels, 300, kernel_size=3, stride=1, padding='same')
        # self.bn1 = torch.nn.BatchNorm1d(300)  # TODO: norm along an axis or 2D only?

        self.activation1 = torch.nn.ReLU(inplace=True)
        # self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = torch.nn.Conv1d(300, self.output_channels, kernel_size=1, stride=1, padding='same')
        self.activation2 = torch.nn.ReLU(inplace=True)
        # self.layer2 = torch.nn.Conv2d(in_channels=6, out_channels=1, kernel_size=(6, 5), stride=1,
        #                               padding='same')

    def forward(self, x):
        batch, xis, xs, ys = x.shape
        x = x.reshape((batch, xis, -1))
        x = self.layer1(x)
        # x = self.bn1(x)
        x = self.activation1(x)
        # x = self.pool1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        # print(x.shape)
        x = x.reshape((batch, self.output_channels, xs, ys))
        return x


class INFERDefaultModel(torch.nn.Module):
    """
    A simple 2 layer fcn to be used as a default fallback model when no standard model is provided.
    """

    def __init__(self, window_size, output_channels):
        super(INFERDefaultModel, self).__init__()
        self.window_size = window_size
        self.output_channels = output_channels
        # self.layer1 = torch.nn.Linear(in_features=3 * self.window_size * self.window_size * batch, out_features=400) # limited for size
        # self.activation1 = torch.nn.ReLU(inplace=True)
        # self.dropout1 = torch.nn.Dropout(0.25)
        # self.layer2 = torch.nn.Linear(in_features=400, out_features=self.output_channels)
        self.layer1 = torch.nn.Conv2d(3, out_channels=self.output_channels, kernel_size=3, stride=1,
                                      padding=1)  # limited for size
        self.activation1 = torch.nn.Sigmoid()

    def forward(self, x):
        # print(f"input: {x.shape}")
        x = self.layer1(x)
        x = self.activation1(x)
        # print(f"output: {x.shape}")
        return x


class CombinedModel(torch.nn.Module):
    """
    This model is made for the purpose of using 3 dimensional data as inputs
    where the last 2 dimensions are X and Y axes of images.

    concatenating with other publicly available models.
    This model is not intented to be used on its own

    """

    def __init__(self, input_shape, segmentation_model, n_classes, window_size=200, batchsize=80):
        """
        window size is only relevant for when there is no
        """
        super(CombinedModel, self).__init__()
        self.models = []
        self.window_size = window_size
        self.feature_extractor = INFERFeatureExtractor(input_shape)
        self.input_channels = self.feature_extractor.input_channels
        self.outputchannels = n_classes

        if segmentation_model is not None:
            self.segmentation_model = segmentation_model
        else:
            self.segmentation_model = INFERDefaultModel(self.window_size, self.outputchannels, batch=batchsize)

        # print(f"Generated Combined model: {CombinedModel} "
        #       f"using FeatureExtractor: {self.feature_extractor} and Segmentation Model: {self.segmentation_model}")

    def forward(self, x):
        features = self.feature_extractor(x)
        # segmentation_output = None
        segmentation_output = self.segmentation_model(features)
        try:
            segmentation_output = segmentation_output['out']
        except:
            segmentation_output = segmentation_output.squeeze()
            # segmentation_output = segmentation_output.type(torch.cuda.LongTensor)  # TODO
            # print("segmentation output:", segmentation_output.shape, type(segmentation_output))
        # concat_output = torch.cat((features, segmentation_output), dim=1)
        # concat_output = self.concat_layer(concat_output)
        # final_output = self.final_layer(concat_output)
        return segmentation_output
        # return x


class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        # self.drop = torch.nn.Dropout()
        self.conv = torch.nn.Conv2d(self.hidden_size, self.num_classes, kernel_size=1, stride=1, padding='same')

    def forward(self, x):
        # x = (batch, xi, x, y) --> (xi, # TODO: is dim permutation required?
        # x = x.permute(0, 2, 1, 3)
        batchsize, xis, xs, ys = x.size()  # torch size not np
        x = x.permute(0, 2, 3, 1)
        x = x.reshape((batchsize, -1, xis))
        ###################################################################################
        # x = x.permute(1, 0, 2, 3)  # bring all dimensions except xi together
        # x = x.reshape((xis, -1))  # combine those dimensions
        # x = x.permute((1, 0))  # bring batch back to the front
        # x = x.reshape(x.shape+(self.input_size,))
        ###################################################################################
        # print("XSIZE: ", x.size())
        # h0 = torch.zeros(self.num_layers, batchsize, self.hidden_size).to(x.device)
        # c0 = torch.zeros(self.num_layers, batchsize, self.hidden_size).to(x.device)
        # input (x) should be of dims : (batchsize, seq. length, input_size)
        h0 = torch.zeros(self.num_layers, batchsize, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batchsize, self.hidden_size).to(x.device)
        x, (hn, cn) = self.lstm(x, (h0, c0))  # xshape = (newbatchsize, xis, hiddensize)
        # print(x.shape, hn.shape, cn.shape)
        x = x.permute(0, 2, 1)
        x = x.reshape(batchsize, self.hidden_size, xs, ys)
        # print(x.shape)
        # x = x.reshape(batchsize, xs, ys, self.hidden_size, xis, 1)
        x = self.conv(x)
        return x


class GRUModel(torch.nn.Module):  # TODO
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_layers = num_layers

        self.gru = torch.nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        # self.drop = torch.nn.Dropout()
        # downsampling
        self.down_conv = torch.nn.Conv2d(self.hidden_size, out_channels=16, kernel_size=3, padding=1, stride=2)
        # upsampling
        self.up_conv = torch.nn.ConvTranspose2d(in_channels=16, out_channels=self.hidden_size, kernel_size=22, stride=2)
        self.direct_conv = torch.nn.Conv2d(self.hidden_size, self.num_classes, kernel_size=1, stride=1, padding='same')

    def forward(self, x):
        batchsize, xis, xs, ys = x.size()  # torch size not np
        x = x.permute(0, 2, 3, 1)  # TODO x.view
        x = x.reshape((batchsize, -1, xis))
        x, hn = self.gru(x)  # xshape = (newbatchsize, xis, hiddensize)
        x = x.permute(0, 2, 1)
        x = x.reshape(batchsize, self.hidden_size, xs, ys)
        downsampled = self.down_conv(x)
        downsampled = torch.nn.functional.relu(downsampled)
        downsampled = self.up_conv(downsampled)
        direct = self.conv(x)
        x = downsampled + direct
        return x


class GRU_3D_Model(torch.nn.Module):  # TODO
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRU_3D_Model, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_layers = num_layers

        self.gru = torch.nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        # self.drop = torch.nn.Dropout()
        # downsampling
        self.down_conv = torch.nn.Conv3d(self.hidden_size, out_channels=16, kernel_size=3, padding=1, stride=2)
        # upsampling - the padding = 1 for odd dimensions. So it might be best to have all even dimensions.
        self.up_conv = torch.nn.ConvTranspose3d(in_channels=16, out_channels=self.hidden_size, kernel_size=6, stride=2,
                                                padding=2)

        self.direct_conv = torch.nn.Conv3d(self.hidden_size, self.hidden_size, kernel_size=1, stride=1, padding='same')
        self.final_conv = torch.nn.Conv3d(self.hidden_size * 2, self.num_classes, kernel_size=1, stride=1,
                                          padding='same')

    def forward(self, x):
        batchsize, xis, zs, xs, ys = x.size()  # torch size not np
        # print(x.shape, flush=True)
        # print(x)
        # x = x.permute(0, 2, 3, 4, 1)  # TODO x.view
        # print(x.shape, flush=True)
        # print(x)
        # exit()
        x = x.view(batchsize, xis, -1)  # TODO x.view
        x = x.permute(0, 2, 1)
        # print(x.shape, flush=True)

        # x = x.reshape((batchsize, -1, xis))
        x, hn = self.gru(x)  # xshape = (newbatchsize, xis, hiddensize)
        # x = x.reshape(batchsize, zs, xs, ys, self.hidden_size)
        # print(x.shape, flush=True)
        x = x.permute(0, 2, 1)  # bring back original order
        x = x.reshape(batchsize, self.hidden_size, zs, xs, ys)
        # x = x.permute(0, 2, 1)
        # x = x.reshape(batchsize, self.hidden_size, xs, ys)
        downsampled = self.down_conv(x)
        downsampled = torch.nn.functional.relu(downsampled)
        upsampled = self.up_conv(downsampled)
        direct = self.direct_conv(x)
        # print(upsampled.shape, direct.shape, downsampled.shape)
        # print(batchsize, xis, zs, xs, ys)
        x = torch.cat([upsampled, direct], dim=1)
        return x


class TomographyModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__(TomographyModel, self).__init__()
        self.output_layer = torch.nn.Linear()

    def forward(self, x):
        self.output_layer = torch.nn.Linear()


class FCNModel(torch.nn.Module):
    def __init__(self):
        super(FCNModel, self).__init__()
        self.layer1 = torch.nn.Linear(in_features=3, out_features=4)
        self.layer2 = torch.nn.Linear()
        self.layer3 = torch.nn.Linear()
        self.output_layer = torch.nn.Linear()

    def forward(self, x):
        return x


class SquishModel(torch.nn.Module):
    def __init__(self):
        super(SquishModel, self).__init__()
        self.layer1 = torch.nn.Linear()
        self.layer2 = torch.nn.Linear()
        self.layer3 = torch.nn.Linear()
        self.output_layer = torch.nn.Linear()

    def forward(self, x):
        print(x.shape)
        return x


if __name__ == "__main__":
    """
    Example architecture: INFER_Model_cn1 + Deeplab50
    
    To incorporate autocorrelation and other parameters the following options seemto be  worth looking into:
        1. ensembling/ combining the segmentation output with a small network that accepts the output 
        from combined net image parameters
        2. multi input for first block : https://rosenfelder.ai/multi-input-neural-network-pytorch/
    """
    # from torchvision import models
    #
    outputchannels = 255
    input_channels = 168
    image_shape = (input_channels, 2000, 2000, 2000)
    window = 200
    # infer_input_model = LSTMModel(input_size=input_channels, hidden_size=32, num_layers=2,
    #                               num_classes=outputchannels)
    infer_input_model = GRU_3D_Model(input_size=input_channels, hidden_size=64, num_layers=2,
                                     num_classes=outputchannels)

    # infer_input_model = INFERFeatureExtractor1D(input_channels=input_channels, output_channels=outputchannels)
    # infer_input_model = CombinedModel(input_channels, None, outputchannels, window_size=200, batchsize=80)
    print(summaryX(infer_input_model, torch.zeros((80, 50, window, window, window))))
    # print(summary(infer_input_model, (252, window, window), batch_size=80))
    # preset_model = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=outputchannels,
    #                                                       progress=True)
    # combined_model = CombinedModel(input_shape=input_channels, segmentation_model=preset_model,
    #                                n_classes=outputchannels, window_size=200)
    # print(summary(infer_input_model, (252, 96, 96)))
    # # print(summary(combined_model, (252, 96, 96)))
    # # exit()
    # import numpy as np
    #
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(combined_model.parameters(), lr=0.001)
    # num_epochs = 1
    # combined_model.train()
    # for epoch in range(num_epochs):
    #     input_data = torch.from_numpy(np.random.random((2, 252, 96, 96)).astype(np.float32))
    #     target = torch.from_numpy((np.random.random((2, outputchannels, 96, 96)) > 0.3).astype(np.float32))
    #     outputs = combined_model(input_data)
    #     loss = criterion(outputs, target)
    #     # Backward pass and optimization
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    # encoded = infer_input_model(input)
    # output = modelB(x2)
    # augmented_preset_model = CombinedModel().combine([infer_input_model, preset_model])
    # augmented_preset_model = torch.cat([infer_input_model, preset_model], dim=1)
    # preset_model.detach() if weights not to be changed
    """
    Necessary to fix conv1d not working
    !pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
    """
