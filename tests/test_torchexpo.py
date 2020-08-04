"""torchexpo tests"""
import torch
import torchvision.models as models
from torchexpo.core import TorchExpo

def test_get_extracted_file_name():
    """Test Get Extracted File Name"""

    model = TorchExpo(model=get_test_model(), model_name="FCN ResNet-18",
                      model_example=get_test_model_example())
    assert model.get_extracted_file_name() == "fcn_resnet18"

    model = TorchExpo(model=get_test_model(), model_name="GoogLeNet",
                      model_example=get_test_model_example())
    assert model.get_extracted_file_name() == "googlenet"

    model = TorchExpo(model=get_test_model(), model_name="Mask-R_CNN",
                      model_example=get_test_model_example())
    assert model.get_extracted_file_name() == "maskr_cnn"

def get_test_model():
    """Returns dummy model for arguments"""
    return models.resnet18()

def get_test_model_example():
    """Returns dummy model example for arguments"""
    return torch.rand([1, 1])
