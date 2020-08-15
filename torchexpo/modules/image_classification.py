import torch
from torch.utils import mobile_optimizer
from torchexpo.modules import TorchExpoModule


class ImageClassificationModule(TorchExpoModule):
    """
    Image Classification module for all vision models under vision/image_classification

    Args:
        model: Pretrained model which is used for conversion
        model_name: Name of the model
        model_example: Example tensor used as example input while extraction
    """
    def __init__(self, model, model_name, model_example):
        if model_example == 'default':
            model_example = torch.rand(1, 3, 224, 224)
        super(ImageClassificationModule, self).__init__(model, model_name, model_example)

    def extract_torchscript(self):
        """Extract torchscript module from image classification model"""
        super().print_message("torchscript")
        scripted_module = torch.jit.script(self.model)
        optimized_module = mobile_optimizer.optimize_for_mobile(scripted_module)
        optimized_module.save("{}.pt".format(self.file_name))

    def extract_onnx(self, opset_version=None):
        """
        Extract onnx from image classification model

        Args:
            opset_version: Opset version used while ONNX conversion
        """
        super().print_message("onnx")
        torch.onnx.export(self.model, self.model_example, "{}.onnx".format(self.file_name))