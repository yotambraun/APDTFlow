from setuptools import setup, find_packages
import os

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="apdtflow",
    version="0.2.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchdiffeq",
        "pandas",
        "numpy",
        "matplotlib",
        "tensorboard",
        "pyyaml",
        "scikit-learn",
        "statsmodels",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    description="APDTFlow: A modular forecasting framework for time series data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Yotam Barun",
    author_email="yotambarun93@gmail.com",
    url="https://github.com/yotambraun/APDTFlow",
    project_urls={
        "Homepage": "https://github.com/yotambraun/APDTFlow",
        "Documentation": "https://github.com/yotambraun/APDTFlow",
        "Source": "https://github.com/yotambraun/APDTFlow",
    },
    python_requires=">=3.10",
    entry_points={
        "console_scripts": ["apdtflow=apdtflow.cli:main"],
    },
)
