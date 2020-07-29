"""Mask R-CNN Model"""
import torch
import torchvision
from torchexpo import TorchExpo


mask_rcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
mask_rcnn.eval()
example = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]

TorchExpo(model_name='Mask R-CNN', model=mask_rcnn, model_example=example, extract_format='onnx')
