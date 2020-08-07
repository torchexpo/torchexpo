import torchexpo
from torchexpo.vision import image_segmentation


print(torchexpo.__version__)
fcn = image_segmentation.fcn_resnet101()
fcn.extract_torchscript()
fcn.extract_onnx()
