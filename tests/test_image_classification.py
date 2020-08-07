from torchexpo.vision import image_classification


def test_alexnet():
    """Test AlexNet"""
    alexnet = image_classification.alexnet()
    alexnet.extract_torchscript()
    alexnet.extract_onnx()

def test_googlenet():
    """Test GoogLeNet"""
    googlenet = image_classification.googlenet()
    googlenet.extract_torchscript()
    googlenet.extract_onnx()

def test_densenet():
    """Test DenseNet"""
    densenet = image_classification.densenet()
    densenet.extract_torchscript()
    densenet.extract_onnx()

def test_inception():
    """Test Inception"""
    inception = image_classification.inception()
    inception.extract_torchscript()
    inception.extract_onnx()

def test_mnasnet():
    """Test MNASNet"""
    mnasnet = image_classification.mnasnet()
    mnasnet.extract_torchscript()
    mnasnet.extract_onnx()

def test_mobilenet():
    """Test MobileNet"""
    mobilenet = image_classification.mobilenet()
    mobilenet.extract_torchscript()
    mobilenet.extract_onnx()

def test_resnet18():
    """Test ResNet18"""
    resnet18 = image_classification.resnet18()
    resnet18.extract_torchscript()
    resnet18.extract_onnx()

def test_shufflenet():
    """Test ShuffleNet"""
    shufflenet = image_classification.shufflenet()
    shufflenet.extract_torchscript()
    shufflenet.extract_onnx()

def test_vgg16():
    """Test VGG16"""
    vgg16 = image_classification.vgg16()
    vgg16.extract_torchscript()
    vgg16.extract_onnx()