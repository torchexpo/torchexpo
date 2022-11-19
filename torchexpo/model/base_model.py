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
        self.model = None
        self.derived_task = BaseTask
        self.initialized_model = None

    @property
    def name(self):
        """Model name"""
        return self.model["name"]

    @property
    def publisher(self):
        """Publisher name"""
        return self.model["publisher"]["name"]

    @property
    def task(self):
        """Task name"""
        return self.model["task"]["name"]

    @property
    def dataset(self):
        """Dataset name"""
        return self.model["dataset"]["name"]

    @property
    def filename(self):
        """Loaded module name"""
        return self.model['slug']+'.pt'

    def fetch_model(self):
        """Fetch model"""
        res = requests.get(f"{os.getenv('API_URL', API_URL)}/graphql",
                           params={"query": MODEL_QUERY.format(slug=self.slug)})
        res.raise_for_status()
        res_json = res.json()
        self.model = res_json['data']['model']
        self.derived_task = derive_task(self.model['task']['slug'])

    def initialize_model(self):
        """Initialize model"""
        # download the url
        _urlretrieve(self.model['download'], self.filename)
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
