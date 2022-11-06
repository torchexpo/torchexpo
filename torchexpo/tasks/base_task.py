"""Base Task"""
from typing import Any
from abc import ABC, abstractmethod


class BaseTask(ABC):
    """TorchExpo Base Task"""

    def __init__(self) -> None:
        pass

    @abstractmethod
    def preprocess(self):
        """Preprocess"""
        raise NotImplementedError

    def inference(self):
        """Inference"""
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, model_output: Any, topk: int, map_class_to_label: bool = False):
        """Postprocess"""
        raise NotImplementedError
