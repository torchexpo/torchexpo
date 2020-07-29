"""Mask R-CNN Model"""
import torch
import torchvision
from torchexpo.core.torchexpo import TorchExpo


class MaskRCNN(TorchExpo):
    """Mask R-CNN Model"""

    name = "Mask R-CNN"
    example = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]

    def __init__(self):
        """Initialize Model"""
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        super().__init__(self.model, self.name, self.example)

        self.file_name = self.get_extracted_file_name()

    def extract_torchscript(self):
        traced_script_module = torch.jit.trace(self.model, self.example)
        traced_script_module.save("{}.pt".format(self.file_name))
