from setuptools import setup, find_packages
import os

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="apdtflow",
    version="0.3.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchdiffeq",
        "pandas",
        "numpy",
        "matplotlib",
        "scipy",
        "tensorboard",
        "pyyaml",
        "scikit-learn",
        "statsmodels",
        "tqdm",
    ],
    keywords=[
        "time-series",
        "forecasting",
        "neural-ode",
        "deep-learning",
        "transformer",
        "tcn",
        "timeseries-analysis",
        "time-series-forecasting",
        "neural-differential-equations",
        "probabilistic-forecasting",
        "uncertainty-quantification",
        "conformal-prediction",
        "exogenous-variables",
        "multi-scale-decomposition",
        "pytorch",
        "machine-learning",
        "prophet-alternative",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
    description="APDTFlow: A modular forecasting framework for time series data with Neural ODEs, Conformal Prediction, and Exogenous Variables",
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
