import torch
from torch.utils import mobile_optimizer
from torchexpo.modules import TorchExpoModule


class ImageClassificationModule(TorchExpoModule):
    """
    Image Classification module for all vision models under vision/image_classification
    """
    def __init__(self, model, model_name, model_example=torch.rand(1, 3, 224, 224)):
        super(ImageClassificationModule, self).__init__(model, model_name, model_example)

    def extract_torchscript(self):
        super().print_message("torchscript")
        scripted_module = torch.jit.script(self.model)
        optimized_module = mobile_optimizer.optimize_for_mobile(scripted_module)
        optimized_module.save("{}.pt".format(self.file_name))

    def extract_onnx(self, opset_version=None):
        super().print_message("onnx")
        torch.onnx.export(self.model, self.model_example, "{}.onnx".format(self.file_name))