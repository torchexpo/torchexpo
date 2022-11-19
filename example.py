import torchexpo
from PIL import Image
from torchvision.models import EfficientNet_B0_Weights
from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights


example = Image.open(
    "../../personal/bitbeast/pytorch-grpc-serving/model/example.jpg")

ic_weights = EfficientNet_B0_Weights.IMAGENET1K_V1
preprocess = ic_weights.transforms()
ic = torchexpo.model(slug="torchexpo-efficientnet-b0-imagenet1k-v1")
output = ic.inference(preprocess(example).unsqueeze(0))
print(ic.postprocess(model_output=output, topk=5,
      map_class_to_label=ic_weights.meta["categories"]))

ss_weights = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
preprocess = ss_weights.transforms()
ss = torchexpo.model(
    slug="torchexpo-deeplabv3-mobilenet-v3-large-coco-with-voc-labels-v1")
output = ss.inference(preprocess(example).unsqueeze(0))
print(ss.postprocess(model_output=output, topk=5,
      map_class_to_label=ss_weights.meta["categories"]))

od_weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
preprocess = od_weights.transforms()
od = torchexpo.model(
    slug="torchexpo-fasterrcnn-mobilenet-v3-large-fpn-coco-v1")
output = od.inference(preprocess(example))
print(od.postprocess(model_output=output, topk=5,
      map_class_to_label=od_weights.meta["categories"]))
