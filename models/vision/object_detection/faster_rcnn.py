"""Faster R-CNN Model"""
import torch
import torchvision
from torchexpo import TorchExpo


faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
faster_rcnn.eval()
example = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]

TorchExpo(model_name='Faster R-CNN', model=faster_rcnn, model_example=example, extract_format='onnx')
