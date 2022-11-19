"""TorchExpo Model"""
from torchexpo.model.base_model import BaseModel


def model(slug: str, initialize: bool = True) -> BaseModel:
    """Driver function"""
    base_model = BaseModel(slug=slug)
    base_model.fetch_model()
    if initialize:
        base_model.initialize_model()
    return base_model
