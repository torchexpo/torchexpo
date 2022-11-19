"""Base Task"""
from typing import Any, List
from abc import ABC, abstractmethod


class BaseTask(ABC):
    """TorchExpo Base Task"""

    def __init__(self) -> None:
        pass

    @abstractmethod
    def preprocess(self):
        """Preprocess"""
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, model_output: Any, topk: int, map_class_to_label: List[str]) -> Any:
        """Postprocess"""
        raise NotImplementedError
