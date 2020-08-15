class TorchExpoModule:
    """
    Base class for all PyTorch models

    Args:
        model: Pretrained model which is used for conversion
        model_name: Name of the model
        model_example: Example tensor used as example input while extraction
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
        """
        Prints message

        Args:
            output_type: Name of the output format used for printing
        """
        print("Extracting model {} in {} format".format(self.model_name, output_type))

    def extract_onnx(self, opset_version=None):
        """
        Extracts model in ONNX format

        Args:
            opset_version: Opset version used while ONNX conversion
        """
        raise NotImplementedError

    def extract_caffe2_mobile(self):
        """Extracts model in Caffe2 Mobile format"""
        self.print_message("caffe2 mobile")
        raise NotImplementedError

    def extract_torchscript(self):
        """Extracts model in TorchScript format"""
        raise NotImplementedError