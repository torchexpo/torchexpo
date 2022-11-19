# Contribution Guide

## Contributing Models

### Creating TorchScript Module

- Follow the [Tutorial](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) for creating a TorchScript module. Every model should have a `forward` method for running the model inference. Each TorchScript module needs to have the output which should match according to the TorchExpo defined tasks.
- Refer [script_and_publish.py](https://github.com/torchexpo/torchvision-models/blob/master/script_and_publish.py) as an example for generating the module and publishing to GitHub.
- Output of the TorchScript module should match as follows:
  - **Computer Vision**:
    - **Image Classification**: `torch.Tensor` `([N])`
    - **Semantic Segmentation**: `torch.Tensor` `([N][H][W])`
    - **Object Detection**: `[{"boxes": torch.Tensor, "labels": torch.Tensor, "scores": torch.Tensor}]` `({"key": [N]})`
  - **Natural Language Processing**:
    - ` Coming Soon`
- Save the scripted module as `{publisherSlug}-{modelSlug}.pt` e.g. `torchexpo-convnext-base-imagenet1k-v1.pt`

### Releasing & Hosting TorchScript Module

- Once the TorchScript module is scripted and saved as `.pt` file. One needs to host and make it available as a public download url. Best way is to create a GitHub release and add scripted module as release assets.
- Refer [GaAMA example](https://github.com/prabhuomkar/bitbeast/blob/master/gaama/examples/example.ipynb) for creating a release and publishing release assets in Python.

### Creating Pull Request to TorchExpo

- Once your scripted module is open sourced, your module needs to be added in the TorchExpo website. Create a Pull Request to [torchexpo repository](https://github.com/torchexpo/torchexpo) with the data of your model.
- Create PR with following files:
  - Model File: `data/models/{publisherSlug}/models/{modelSlug}/{modelSlug}.yaml`
  - Model Description File: `data/models/{publisherSlug}/models/{modelSlug}/{modelSlug}.md`
  - Publisher File: `data/models/{publisherSlug}/{publisherSlug}.yaml`
- Refer [data folder](https://github.com/torchexpo/torchexpo/tree/master/data) for examples on the structure of the data that needs to be contributed.

## Contributing Collections

TBD
