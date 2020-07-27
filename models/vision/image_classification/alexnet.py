"""AlexNet Model"""
import torch
import torchvision
from torchexpo import TorchExpo


alexnet = torchvision.models.alexnet(pretrained=True)
alexnet.eval()
example = torch.rand(1, 3, 224, 224)

TorchExpo(model_name='AlexNet', model=alexnet, model_example=example, extract_format='torchscript')
