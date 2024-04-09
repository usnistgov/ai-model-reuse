import torch
# import numpy as np


# activation = {}
#
#
def get_layer_output(layer_outputs=None):
    # if activation is None:
    #     activation = {}

    def hook(model, input, output):
        # activation
        # activation.clear()  # remove all previous values
        # print(activation)
        layer_out = torch.squeeze(output, 0)
        layer_out = layer_out.cpu().detach().numpy()
        layer_out = layer_out.transpose((1, 2, 0))
        layer_outputs.append(layer_out)  # remember only last value
        # print(activation)

    return hook
#
#
# captured_outputs = []
#
#
# def capture_output(module, input, output):
#     captured_outputs.append(output.detach())


def get_layerweights(modelname, layer_batchlist=None, n=1):
    if layer_batchlist is None:
        layer_batchlist = []
    # assert n >= 1, "Layer must be greater than or equal to 1"
    # layer = next(model.named_children())
    # if n > 1:
    #     for n in range(n):
    #         layer = next(model.named_children())
    # print(layer)
    # layer.register_forward_hook(get_activation('feature_extractor'))
    # name = 'feature_extractor'
    modelname.feature_extractor.register_forward_hook(get_layer_output(layer_outputs=layer_batchlist))
    # print("activation", activation)
    # print("layers", len(layers))
    # try:
    #     print("layers", layers[-1].shape)
    # except:
    #     print("layers failed")
    return layer_batchlist  # return and remove last and only element in list
    # model.feature_extractor.layer2.register_forward_hook(capture_output)
    # print(len(captured_outputs))
    # print(captured_outputs[-1])


# def clear_outputs():
#     captured_outputs.clear()


if __name__ == "__main__":
    testmodel = torch.model()  # replace with actual model
    testmodel.layername.register_forward_hook(get_layer_output('fc2'))
    x = torch.randn(1, 25)
    output = testmodel(x)
    # print(activation['fc2'])
