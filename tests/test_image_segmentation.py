# from torchexpo.vision import image_segmentation


# def test_fcn_resnet():
#     """Test FCN-ResNet"""
#     fcn_resnet = [image_segmentation.fcn_resnet50()]
#     map(extract_image_segmentation, fcn_resnet)

# def test_deeplabv3_resnet():
#     """Test DeepLabV3-ResNet"""
#     deeplabv3_resnet = [image_segmentation.deeplabv3_resnet50(),
#                         image_segmentation.deeplabv3_resnet101()]
#     map(extract_image_segmentation, deeplabv3_resnet)

# def extract_image_segmentation(model):
#     """Runs extraction common for all image segmentation models"""
#     model.extract_torchscript()
#     # model.extract_onnx()