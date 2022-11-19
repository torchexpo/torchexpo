"""Image Classification"""
from typing import Any, Dict, List

import torch

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
                    map_class_to_label: List[str]) -> Dict[str, Any]:
        """Postprocess Output of Image Classification"""
        top_prob, top_class = torch.topk(model_output, topk)
        return dict({
            map_class_to_label[top_class[0]
                               [idx].item()]: top_prob[0][idx].item()
            for idx in range(0, topk)
        })
