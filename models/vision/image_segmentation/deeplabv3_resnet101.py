"""DeepLabV3-ResNet101 Model"""
import torch
from torchexpo import TorchExpo


deeplabv3 = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
deeplabv3.eval()
example = torch.rand(1, 3, 224, 224)

TorchExpo(model_name='DeepLabV3-ResNet101',
          model=deeplabv3,
          model_example=example,
          extract_format='torchscript')
