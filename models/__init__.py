from . import resnet
from . import vgg
import  torchvision.models
def build_model(model,num_classes=1000,pretraind=True):
    if 'resnet34' in model:
        model = resnet.resnet34(pretrained=pretraind,num_classes=num_classes)
    elif 'resnet50' in model:
        model = resnet.resnet50(pretrained=pretraind,num_classes=num_classes)
    elif 'vgg16' in model:
        model = vgg.vgg16_bn(pretrained=pretraind, num_classes=num_classes)
    else:
        raise ValueError(model)
    return model