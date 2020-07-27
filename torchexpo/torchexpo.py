"""TorchExpo"""
import torch
import torchvision


class TorchExpo:
    """TorchExpo"""
    def __init__(self, model_name='', model=None, model_example=None, extract_format=''):
        if extract_format not in ['onnx', 'torchscript']:
            raise Exception('extraction format can be onnx or torchscript')
        if not model_name:
            raise Exception('model name cannot be empty')
        if model is None:
            raise Exception('model is required')
        if model_example is None:
            raise Exception('model example is required')

        self.model_name = model_name
        self.model = model
        self.model_example = model_example
        self.extract_format = extract_format

        print(torch.__version__)
        print(torchvision.__version__)

        if self.extract_format == 'onnx':
            self.extract_onnx()
        else:
            self.extract_torchscript()

    def get_extracted_file_name(self):
        """Returns file name for output"""
        return self.model_name.lower().replace(' ', '_').replace('-', '')

    def extract_onnx(self):
        """Extracts model in ONNX format"""
        print('Extracting model {} in onnx format'.format(self.model_name))
        raise NotImplementedError

    def extract_torchscript(self):
        """Extracts model in TorchScript format"""
        print('Extracting model {} in torchscript format'.format(self.model_name))
        traced_script_module = torch.jit.trace(self.model, self.model_example)
        traced_script_module.save('{}.pt'.format(self.get_extracted_file_name()))
