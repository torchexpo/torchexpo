"""DeepLabV3-ResNet101 Model"""
import torch
from torchexpo.core.torchexpo import TorchExpo


class DeepLabV3ResNet101(TorchExpo):
    """DeepLabV3-ResNet101 Model"""

    name = "DeepLabV3-ResNet101"
    example = torch.rand(1, 3, 224, 224)

    def __init__(self):
        """Initialize Model"""
        self.model = torch.hub.load("pytorch/vision:v0.6.0", "deeplabv3_resnet101", pretrained=True)
        super().__init__(self.model, self.name, self.example)

        self.file_name = self.get_extracted_file_name()

    def extract_torchscript(self):
        traced_script_module = torch.jit.trace(self.model, self.example)
        traced_script_module.save("{}.pt".format(self.file_name))
