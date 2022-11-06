"""Model"""
import os
from typing import Any
import requests

from torchexpo.tasks import derive_task, BaseTask


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

    def fetch_model(self):
        """Fetch model"""
        res = requests.post(f"{os.getenv('API_URL', API_URL)}/graphql",
                            data=MODEL_QUERY.format(slug=self.slug))
        res.raise_for_status()
        res_json = res.json()
        model = res_json['data']['model']
        self._name = model['name']
        self._publisher = model['publisher']['name']
        self._task = model['task']['name']
        self._dataset = model['dataset']['name']
        self.derived_task = derive_task(model['task']['slug'])

    def initialize_model(self):
        """Initialize model"""
        raise NotImplementedError

    def preprocess(self, model_input: Any):
        """Preprocess input"""
        raise NotImplementedError

    def inference(self, model_input: Any) -> Any:
        """Inference run"""
        return self.derived_task.inference(model_input)

    def postprocess(self, model_output: Any, topk: int, map_class_to_label: bool = False) -> Any:
        """Postprocess output"""
        return self.derived_task.postprocess(
            model_output, topk=topk, map_class_to_label=map_class_to_label)
