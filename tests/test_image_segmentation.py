from torchexpo.vision import image_segmentation


def test_fcn_resnet101():
    """Test FCN-ResNet101"""
    fcn_resnet101 = image_segmentation.fcn_resnet101()
    fcn_resnet101.extract_torchscript()
    # fcn_resnet101.extract_onnx()

def test_deeplabv3_resnet101():
    """Test DeepLabV3-ResNet101"""
    deeplabv3_resnet101 = image_segmentation.deeplabv3_resnet101()
    deeplabv3_resnet101.extract_torchscript()
    # deeplabv3_resnet101.extract_onnx()