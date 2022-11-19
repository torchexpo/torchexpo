"""TorchExpo Tasks"""
from typing import Any

from torchexpo.tasks.vision import ImageClassification, SemanticSegmentation, ObjectDetection


SUPPORTED_TASKS = {
    "image-classification": ImageClassification(),
    "semantic-segmentation": SemanticSegmentation(),
    "object-detection": ObjectDetection(),
}


def derive_task(task_slug: str) -> Any:
    """Derive task from name"""
    return SUPPORTED_TASKS.get(task_slug)
