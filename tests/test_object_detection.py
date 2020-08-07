from torchexpo.vision import object_detection


def test_mask_rcnn():
    """Test Mask R-CNN"""
    mask_rcnn = object_detection.mask_rcnn()
    mask_rcnn.extract_torchscript()
    # mask_rcnn.extract_onnx()