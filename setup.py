import os
from setuptools import setup, find_packages


version = "2.0.0-alpha1"
package_name = "torchexpo"

cwd = os.path.dirname(os.path.abspath(__file__))


def write_version_file():
    version_path = os.path.join(cwd, package_name, "version.py")
    with open(version_path, "w") as f:
        f.write("__version__ = '{}'\n".format(version))


write_version_file()

with open(os.path.join(cwd, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

install_requires = [
    "torch",
    "torchvision"
]

setup(
    name=package_name,
    version=version,
    description="Collection of models and extensions for deployment in PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/torchexpo/torchexpo",
    download_url="https://github.com/torchexpo/torchexpo/tags",
    author="Omkar Prabhu",
    author_email="prabhuomkar@pm.me",
    license="Apache-2.0",
    packages=find_packages(
        exclude=(".github", "tests", "docs", "examples", "scripts")),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
    ],
    zip_safe=True,
    install_requires=install_requires,
)
