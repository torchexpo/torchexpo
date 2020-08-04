import os
import subprocess
from setuptools import setup, find_packages


version = "1.0.0-alpha2"
sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8") 
package_name = "torchexpo"

cwd = os.path.dirname(os.path.abspath(__file__))

def write_version_file():
    version_path = os.path.join(cwd, package_name, "version.py")
    with open(version_path, "w") as f:
        f.write("__version__ = '{}'\n".format(version))
        f.write("git_version = {}\n".format(repr(sha)))

write_version_file()

readme = open("README.md").read()

requirements = open("requirements.txt").read().split()

setup(
    name=package_name,
    version=version,
    author="Omkar Prabhu",
    author_email="prabhuomkar@pm.me",
    url="https://github.com/torchexpo/torchexpo",
    description="Collection of models and extensions for mobile deployment in PyTorch",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    packages=find_packages(exclude=("tests",)),
    zip_safe=True,
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)