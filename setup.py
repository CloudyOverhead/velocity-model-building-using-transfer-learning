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
        version="eef941d4e31b5fa0dc7823e491e0575ad1e1f423",
    ),
    GITHUB_REQUIREMENT.format(
        author="gfabieno",
        name="ModelGenerator",
        version="a01cbb93d67a6c859ea6ac67ea55361208b66e7c",
    ),
    GITHUB_REQUIREMENT.format(
        author="gfabieno",
        name="Deep_2D_velocity",
        version="f4b0031cae1e3ae21e5eb881af7c7ef22e4f56f0",
    ),
]

setup(
    name="deep-learning-seismic-velocity-estimation",
    version="0.0.1",
    author="Jérome Simon",
    author_email="jerome.simon@ete.inrs.ca",
    description="Estimating 2D seismic velocity models using deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CloudyOverhead/deep-learning-velocity-estimation",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
