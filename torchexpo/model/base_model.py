"""Model"""
import os
from typing import Any
import requests

import torch

from torchexpo.tasks import derive_task, BaseTask

from .utils import _urlretrieve


API_URL = "https://torchexpo.fly.dev"
MODEL_QUERY = """
query {{
    model(slug: "{slug}") {{
        slug
        name
        download
        task {{
            slug
            name
        }}
        publisher {{
            name
        }}
        dataset {{
            name
        }}
    }}
}}
"""


class BaseModel:
    """TorchExpo Model"""

    def __init__(self, slug: str, gpu: bool = False) -> None:
        self.slug = slug
        self.gpu = gpu
        self._name = ''
        self._publisher = ''
        self._task = ''
        self._dataset = ''

        self.derived_task = BaseTask
        self.initialized_model = None

    @property
    def name(self):
        """Model name"""
        return self._name

    @property
    def publisher(self):
        """Publisher name"""
        return self._publisher

    @property
    def task(self):
        """Task name"""
        return self._task

    @property
    def dataset(self):
        """Dataset name"""
        return self._dataset

    @property
    def filename(self):
        """Loaded module name"""
        return self._model['slug']+'.pt'

    def fetch_model(self):
        """Fetch model"""
        res = requests.get(f"{os.getenv('API_URL', API_URL)}/graphql",
                           params={"query": MODEL_QUERY.format(slug=self.slug)})
        res.raise_for_status()
        res_json = res.json()
        self._model = res_json['data']['model']
        self._name = self._model['name']
        self._publisher = self._model['publisher']['name']
        self._task = self._model['task']['name']
        self._dataset = self._model['dataset']['name']
        self.derived_task = derive_task(self._model['task']['slug'])

    def initialize_model(self):
        """Initialize model"""
        # download the url
        _urlretrieve(self._model['download'], self.filename)
        self.initialized_model = torch.jit.load(self.filename)

    def preprocess(self, model_input: Any):
        """Preprocess input"""
        raise NotImplementedError

    def inference(self, model_input: Any) -> Any:
        """Inference run"""
        return self.initialized_model.forward(model_input)

    def postprocess(self, model_output: Any, topk: int, map_class_to_label: bool = False) -> Any:
        """Postprocess output"""
        return self.derived_task.postprocess(
            model_output, topk=topk, map_class_to_label=map_class_to_label)
