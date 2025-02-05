# setup.py
from setuptools import setup, find_packages

setup(
    name='apdtflow',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchdiffeq',
        'pandas',
        'numpy',
        'matplotlib',
        'tensorboard',
        'pyyaml',
    ],
    description='APDTFlow: A modular forecasting framework for time series data',
    author='Yotam Barun',
    author_email='yotambarun93@gmail.com',
    url='https://github.com/yourusername/apdtflow_project',
)
