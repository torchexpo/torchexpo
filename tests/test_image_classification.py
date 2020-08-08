from torchexpo.vision import image_classification


def test_alexnet():
    """Test AlexNet"""
    alexnet = [image_classification.alexnet()]
    map(extract_image_classification, alexnet)

def test_densenet():
    """Test DenseNet"""
    densenet = [image_classification.densenet121(),
                image_classification.densenet161(),
                image_classification.densenet169(),
                image_classification.densenet201()]
    map(extract_image_classification, densenet)

def test_googlenet():
    """Test GoogLeNet"""
    googlenet = [image_classification.googlenet()]
    map(extract_image_classification, googlenet)

def test_inception():
    """Test Inception"""
    inception = [image_classification.inceptionv3()]
    map(extract_image_classification, inception)

def test_mnasnet():
    """Test MNASNet"""
    mnasnet = [image_classification.mnasnet0_5(),
               image_classification.mnasnet0_75(),
               image_classification.mnasnet1_0(),
               image_classification.mnasnet1_3()]
    map(extract_image_classification, mnasnet)

def test_mobilenet():
    """Test MobileNet"""
    mobilenet = [image_classification.mobilenet_v2()]
    map(extract_image_classification, mobilenet)

def test_resnet():
    """Test ResNet"""
    resnet = [image_classification.resnet18(),
              image_classification.resnet34(),
              image_classification.resnet50(),
              image_classification.resnet101(),
              image_classification.resnet152()]
    map(extract_image_classification, resnet)

def test_resnext():
    """Test ResNext"""
    resnext = [image_classification.resnext50_32x4d(),
               image_classification.resnext101_32x8d()]
    map(extract_image_classification, resnext)

def test_shufflenet():
    """Test ShuffleNet"""
    shufflenet = [image_classification.shufflenet_v2_x0_5(),
                  image_classification.shufflenet_v2_x1_0(),
                  image_classification.shufflenet_v2_x1_5(),
                  image_classification.shufflenet_v2_x2_0()]
    map(extract_image_classification, shufflenet)

def test_squeezenet():
    """Test SqueezeNet"""
    squeezenet = [image_classification.squeezenet1_0(),
                  image_classification.squeezenet1_1()]
    map(extract_image_classification, squeezenet)

def test_vgg():
    """Test VGG"""
    vgg = [image_classification.vgg11(),
           image_classification.vgg11_bn(),
           image_classification.vgg13(),
           image_classification.vgg13_bn(),
           image_classification.vgg16(),
           image_classification.vgg16_bn(),
           image_classification.vgg19(),
           image_classification.vgg19_bn()]
    map(extract_image_classification, vgg)

def test_wide_resnet():
    """Test Wide ResNet"""
    wide_resnet = [image_classification.wide_resnet50_2(),
                   image_classification.wide_resnet101_2()]
    map(extract_image_classification, wide_resnet)

def extract_image_classification(model):
    """Runs extraction common for all image classification models"""
    model.extract_torchscript()
    model.extract_onnx()