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
                    map_class_to_label: List[str]) -> List[Dict[str, Any]]:
        """Postprocess Output of Semantic Segmentation"""
        result = []
        for (idx, cls) in enumerate(map_class_to_label):
            mask = model_output[0, idx]
            result.append(
                dict({"label": cls, "score": 0.0, "mask": mask.tolist()}))
        return result
