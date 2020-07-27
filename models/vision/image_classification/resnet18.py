"""ResNet-18 Model"""
import torch
import torchvision
from torchexpo import TorchExpo


resnet18 = torchvision.models.resnet18(pretrained=True)
resnet18.eval()
example = torch.rand(1, 3, 224, 224)

TorchExpo(model_name='ResNet-18',
          model=resnet18,
          model_example=example,
          extract_format='torchscript')
