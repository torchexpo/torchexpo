"""Semantic Segmentation"""
from typing import Any, Dict, List

from torchexpo.tasks.base_task import BaseTask


class SemanticSegmentation(BaseTask):
    """Semantic Segmentation Task"""

    def __init__(self) -> None:
        pass

    def preprocess(self):
        """Preprocess Input for Semantic Segmentation"""
        raise RuntimeError(
            "Preprocess for Semantic Segmentation is not supported")

    def postprocess(self, model_output: Any, topk: int,
                    map_class_to_label: bool = False) -> List[Dict[str, Any]]:
        """Postprocess Output of Semantic Segmentation"""
        return [dict({"label": "", "score": 0.0, "mask": [0.0]})]
