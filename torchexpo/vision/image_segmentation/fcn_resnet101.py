"""FCN-ResNet101 Model"""
import torch
import torchvision
from torchexpo.core.torchexpo import TorchExpo


class FCNResNet101(TorchExpo):
    """FCN-ResNet101 Model"""

    name = "FCN-ResNet101"
    example = torch.rand(1, 3, 224, 224)

    def __init__(self):
        """Initialize Model"""
        self.model = torchvision.models.segmentation.fcn_resnet101(pretrained=True)
        super().__init__(self.model, self.name, self.example)

        self.file_name = self.get_extracted_file_name()

    def extract_torchscript(self):
        traced_script_module = torch.jit.trace(self.model, self.example, strict=False)
        traced_script_module.save("{}.pt".format(self.file_name))
