"""Image Classification"""
from typing import Any, Dict, List

from torchexpo.tasks.base_task import BaseTask


class ImageClassification(BaseTask):
    """Image Classification Task"""

    def __init__(self) -> None:
        pass

    def preprocess(self):
        """Preprocess Input for Image Classification"""
        raise RuntimeError(
            "Preprocess for Image Classification is not supported")

    def postprocess(self, model_output: Any, topk: int,
                    map_class_to_label: bool = False) -> List[Dict[str, Any]]:
        """Postprocess Output of Image Classification"""
        return [dict({"label": "", "score": 0.0})]
