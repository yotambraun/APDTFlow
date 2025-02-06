from setuptools import setup, find_packages
import os

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='apdtflow',
    version='0.1.6',
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
    long_description=long_description, 
    long_description_content_type="text/markdown", 
    author='Yotam Barun',
    author_email='yotambarun93@gmail.com',
    url='https://github.com/yotambraun/APDTFlow',
    project_urls={
        'Homepage': 'https://github.com/yotambraun/APDTFlow',
        'Documentation': 'https://github.com/yotambraun/APDTFlow',
        'Source': 'https://github.com/yotambraun/APDTFlow',
    },
)

