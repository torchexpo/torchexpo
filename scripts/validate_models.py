"""Validate Models"""
import os
import yaml


def get_data(path, slug=True):
    """get data from yaml"""
    contents = yaml.safe_load(open(path).read())
    values = contents["values"]
    if slug:
        return [value["slug"] for value in values]
    return values[0]


def get_datasets():
    """get datasets"""
    return get_data("data/tags/datasets.yaml")


def get_languages():
    """get languages"""
    return get_data("data/tags/languages.yaml")


def get_licenses():
    """get licenses"""
    return get_data("data/tags/licenses.yaml")


def get_tasks():
    """get tasks"""
    return get_data("data/tags/tasks.yaml")


def get_models():
    """get models"""
    result = []
    publishers = os.listdir("data/models")
    for publisher in publishers:
        if os.path.exists(f"data/models/{publisher}/models"):
            models = os.listdir(f"data/models/{publisher}/models")
            for model in models:
                if os.path.isdir(f"data/models/{publisher}/models/{model}"):
                    result.append(
                        get_data(f"data/models/{publisher}/models/{model}/{model}.yaml", False))
    return result


def get_tags():
    """get tags"""
    return {
        "datasets": get_datasets(),
        "languages": get_languages(),
        "licenses": get_licenses(),
        "tasks": get_tasks()
    }


if __name__ == "__main__":
    tags = get_tags()
    models = get_models()
    for model in models:
        if "dataset" in model and not model["dataset"] in tags["datasets"]:
            raise Exception(
                f"model {model['name']} contains invalid dataset {model['dataset']}")
        if "language" in model and not model["language"] in tags["languages"]:
            raise Exception(
                f"model {model['name']} contains invalid language {model['language']}")
        if "license" in model and not model["license"] in tags["licenses"]:
            raise Exception(
                f"model {model['name']} contains invalid license {model['license']}")
        if "task" in model and not model["task"] in tags["tasks"]:
            raise Exception(
                f"model {model['name']} contains invalid task {model['task']}")
