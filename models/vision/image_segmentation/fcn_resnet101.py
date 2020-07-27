"""FCN-ResNet101 Model"""
import torch
from torchexpo import TorchExpo


fcn = torch.hub.load('pytorch/vision:v0.6.0', 'fcn_resnet101', pretrained=True)
fcn.eval()
example = torch.rand(1, 3, 224, 224)

TorchExpo(model_name='FCN-ResNet101',
          model=fcn,
          model_example=example,
          extract_format='torchscript')
