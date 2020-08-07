"""TorchExpo"""
import torch
from torch.utils import mobile_optimizer


class TorchExpo:
    """
    Base class for all PyTorch models
    """
    def __init__(self, model, model_name, model_example):
        if model is None:
            raise Exception("model is required")
        if model_name is None:
            raise Exception("model name cannot be empty")
        if model_example is None:
            raise Exception("model example is required")

        self.model_name = model_name
        self.model = model
        self.model.eval()
        self.model_example = model_example

        self.file_name = self.get_extracted_file_name()

    def get_extracted_file_name(self):
        """Returns file name for output"""
        return self.model_name.lower().replace(" ", "_").replace("-", "")

    def print_message(self, output_type):
        """Prints message"""
        print("Extracting model {} in {} format".format(self.model_name, output_type))

    def extract_onnx(self):
        """Extracts model in ONNX format"""
        self.print_message("onnx")
        torch.onnx.export(self.model, self.model_example, "{}.onnx".format(self.file_name))

    def extract_caffe2_mobile(self):
        """Extracts model in Caffe2 Mobile format"""
        self.print_message("caffe2 mobile")
        raise NotImplementedError

    def extract_torchscript(self):
        """Extracts model in TorchScript format"""
        self.print_message("torchscript")
        scripted_module = torch.jit.script(self.model)
        optimized_module = mobile_optimizer.optimize_for_mobile(scripted_module)
        optimized_module.save("{}.pt".format(self.file_name))