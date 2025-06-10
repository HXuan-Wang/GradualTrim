import torchvision.models as models
import torch.nn as nn
if __name__ == '__main__':
    model_name='resnet34'
    model_fn = getattr(models,model_name)
    model = model_fn(pretrained=True)
    print(model)

    def find_conv_layers(model):
        conv_layers = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.ReLU)):
                conv_layers.append((name))
        return conv_layers


    conv_layers = find_conv_layers(model)
    print(conv_layers)

