"""GoogleNet Model"""
import torch
import torchvision
from torchexpo import TorchExpo


googlenet = torchvision.models.googlenet(pretrained=True)
googlenet.eval()
example = torch.rand(1, 3, 224, 224)

TorchExpo(model_name='GoogleNet',
          model=googlenet,
          model_example=example,
          extract_format='torchscript')
