"""TorchExpo"""
class TorchExpo:
    """TorchExpo"""
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

    def get_extracted_file_name(self):
        """Returns file name for output"""
        return self.model_name.lower().replace(" ", "_").replace("-", "")

    def print_message(self, output_type):
        """Prints message"""
        print("Extracting model {} in {} format".format(self.model_name, output_type))

    def extract_onnx(self):
        """Extracts model in ONNX format"""
        raise NotImplementedError

    def extract_caffe2_mobile(self):
        """Extracts model in Caffe2 Mobile format"""
        raise NotImplementedError

    def extract_torchscript(self):
        """Extracts model in TorchScript format"""
        raise NotImplementedError
