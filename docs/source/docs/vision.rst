torchexpo.vision
################

.. automodule:: torchexpo.vision
  :members:

Image Classification
====================

.. image:: https://res.cloudinary.com/torchexpo/image/upload/v1601144171/assets/tasks/image-classification.jpg
   :width: 25%
   :align: right

|

Image Classification is a fundamental task that attempts to comprehend an entire image as a whole. The goal is to classify the image by assigning it to a specific label. It refers to images in which only one object appears and is analyzed. In contrast, object detection involves both classification and localization tasks, and is used to analyze more realistic cases in which multiple objects may exist in an image.

Example:
    >>> from torchexpo.vision import image_classification
    >>> 
    >>> model = image_classification.squeezenet1_0()
    >>> model.extract_torchscript()
    >>> model.extract_onnx()

.. automodule:: torchexpo.vision.image_classification
  :members:

AlexNet
-------

.. autofunction:: torchexpo.vision.image_classification.alexnet

VGG
---

.. autofunction:: torchexpo.vision.image_classification.vgg11
.. autofunction:: torchexpo.vision.image_classification.vgg11_bn
.. autofunction:: torchexpo.vision.image_classification.vgg13
.. autofunction:: torchexpo.vision.image_classification.vgg13_bn
.. autofunction:: torchexpo.vision.image_classification.vgg16
.. autofunction:: torchexpo.vision.image_classification.vgg16_bn
.. autofunction:: torchexpo.vision.image_classification.vgg19
.. autofunction:: torchexpo.vision.image_classification.vgg19_bn

ResNet
------

.. autofunction:: torchexpo.vision.image_classification.resnet18
.. autofunction:: torchexpo.vision.image_classification.resnet34
.. autofunction:: torchexpo.vision.image_classification.resnet50
.. autofunction:: torchexpo.vision.image_classification.resnet101
.. autofunction:: torchexpo.vision.image_classification.resnet152

SqueezeNet
----------

.. autofunction:: torchexpo.vision.image_classification.squeezenet1_0
.. autofunction:: torchexpo.vision.image_classification.squeezenet1_1

DenseNet
--------

.. autofunction:: torchexpo.vision.image_classification.densenet121
.. autofunction:: torchexpo.vision.image_classification.densenet169
.. autofunction:: torchexpo.vision.image_classification.densenet161
.. autofunction:: torchexpo.vision.image_classification.densenet201

Inception v3
------------

.. autofunction:: torchexpo.vision.image_classification.inceptionv3

GoogLeNet
---------

.. autofunction:: torchexpo.vision.image_classification.googlenet

ShuffleNet v2
-------------

.. autofunction:: torchexpo.vision.image_classification.shufflenet_v2_x0_5
.. autofunction:: torchexpo.vision.image_classification.shufflenet_v2_x1_0

MobileNet v2
------------

.. autofunction:: torchexpo.vision.image_classification.mobilenet_v2

ResNext
-------

.. autofunction:: torchexpo.vision.image_classification.resnext50_32x4d
.. autofunction:: torchexpo.vision.image_classification.resnext101_32x8d

Wide ResNet
-----------

.. autofunction:: torchexpo.vision.image_classification.wide_resnet50_2
.. autofunction:: torchexpo.vision.image_classification.wide_resnet101_2

MNASNet
--------

.. autofunction:: torchexpo.vision.image_classification.mnasnet0_5
.. autofunction:: torchexpo.vision.image_classification.mnasnet1_0

Image Segmentation
==================

.. image:: https://res.cloudinary.com/torchexpo/image/upload/v1601144171/assets/tasks/image-segmentation.jpg
   :width: 25%
   :align: right

|

Image Segmentation (or Semantic Segmentation) is the task of clustering parts of an image together which belong to the same object class. It is a form of pixel-level prediction because each pixel in an image is classified according to a category. Some example benchmarks for this task are Cityscapes, PASCAL VOC and ADE20K. Models are usually evaluated with the Mean Intersection-Over-Union (Mean IoU) and Pixel Accuracy metrics.

Example:
    >>> from torchexpo.vision import image_segmentation
    >>> 
    >>> model = image_segmentation.fcn_resnet50()
    >>> model.extract_torchscript()
    >>> model.extract_onnx()

.. automodule:: torchexpo.vision.image_segmentation
  :members:

FCN-ResNet
----------

.. autofunction:: torchexpo.vision.image_segmentation.fcn_resnet50
.. autofunction:: torchexpo.vision.image_segmentation.fcn_resnet101


DeepLabV3-ResNet
----------------

.. autofunction:: torchexpo.vision.image_segmentation.deeplabv3_resnet50
.. autofunction:: torchexpo.vision.image_segmentation.deeplabv3_resnet101