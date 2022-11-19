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
                    map_class_to_label: List[str]) -> Dict[str, Any]:
        """Postprocess Output of Object Detection"""
        output = model_output[0]
        result = []
        labels = [map_class_to_label[i] for i in output["labels"]]
        for (idx, box) in enumerate(output["boxes"]):
            result.append(dict({
                "label": labels[idx], "score": output["scores"][idx].item(), "box": box.tolist(),
            }))
        return result
