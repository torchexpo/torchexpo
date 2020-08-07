import torch
import torchvision
from torchexpo.core.torchexpo import TorchExpo


def mask_rcnn():
    """Mask R-CNN Model"""
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    obj = TorchExpo(model, "Mask R-CNN", [torch.randn(3, 224, 224), torch.randn(3, 400, 400)])
    return obj