"""TorchExpo Tasks"""
from functools import partial
from typing import Any

from torchexpo.tasks.vision import ImageClassification, SemanticSegmentation, ObjectDetection


SUPPORTED_TASKS = {
    "image-classification": partial(ImageClassification),
    "semantic-segmentation": partial(SemanticSegmentation),
    "object-detection": partial(ObjectDetection),
}


def derive_task(task_slug: str) -> Any:
    """Derive task from name"""
    return SUPPORTED_TASKS.get(task_slug)
