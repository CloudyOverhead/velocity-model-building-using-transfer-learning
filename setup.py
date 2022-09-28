# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

GITHUB_REQUIREMENT = (
    "{name} @ git+https://github.com/{author}/{name}.git@{version}"
)
REQUIREMENTS = [
    GITHUB_REQUIREMENT.format(
        author="gfabieno",
        name="SeisCL",
        version="49595bb9deaf8a4d90e6e2ee09adb378e5ed246e",
    ),
    GITHUB_REQUIREMENT.format(
        author="gfabieno",
        name="ModelGenerator",
        version="v1.0.0",
    ),
    GITHUB_REQUIREMENT.format(
        author="gfabieno",
        name="Deep_2D_velocity",
        version="v1.0",
    ),
    "scikit-image==0.14.2",
    "segyio",
    "scipy",
]

setup(
    name="velocity-model-building-using-transfer-learning",
    version="0.0.1",
    author="Jérome Simon",
    author_email="jerome.simon@ete.inrs.ca",
    description=(
        "Estimating 2D seismic velocity models by leveraging transfer learning"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CloudyOverhead/deep-learning-velocity-estimation",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    setup_requires=['setuptools-git'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
