"""DeepLabV3-ResNet101 Model"""
import abc
import torch
import torch.nn as nn
import torchvision
from torchexpo.core.torchexpo import TorchExpo


class DeepLabV3ResNet101(TorchExpo):
    """DeepLabV3-ResNet101 Model"""

    name = "DeepLabV3-ResNet101"
    example = torch.rand(1, 3, 224, 224)

    def __init__(self):
        """Initialize Model"""
        self.model = TEDeepLabV3ResNet101()
        super().__init__(self.model, self.name, self.example)

        self.file_name = self.get_extracted_file_name()

    def extract_torchscript(self):
        traced_script_module = torch.jit.trace(self.model, self.example)
        traced_script_module.save("{}.pt".format(self.file_name))

class TEDeepLabV3ResNet101(nn.Module):
    """TorchExpo DeepLabV3-ResNet101 Scriptable Module"""
    def __init__(self):
        super(TEDeepLabV3ResNet101, self).__init__()
        self.deeplab = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)

    @abc.abstractmethod
    def forward(self, tensor):
        """Model Forward"""
        output = self.deeplab(tensor)['out']
        return output[0]
