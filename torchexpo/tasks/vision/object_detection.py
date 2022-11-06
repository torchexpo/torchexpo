"""Object Detection"""
from typing import Any, Dict, List

from torchexpo.tasks.base_task import BaseTask


class ObjectDetection(BaseTask):
    """Object Detection Task"""

    def __init__(self) -> None:
        pass

    def preprocess(self):
        """Preprocess Input for Object Detection"""
        raise RuntimeError("Preprocess for Object Detection is not supported")

    def postprocess(self, model_output: Any, topk: int,
                    map_class_to_label: bool = False) -> List[Dict[str, Any]]:
        """Postprocess Output of Object Detection"""
        return [dict({"label": "", "score": 0.0, "box": [0, 0, 0, 0]})]
