# APDTFlow Documentation

Welcome to the APDTFlow forecasting framework documentation. This guide provides a comprehensive overview of the framework, its core components, and how to effectively use and extend it for your time series forecasting tasks.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Installation and Setup](#installation-and-setup)
4. [Getting Started](#getting-started)
    - [Data Loading & Preprocessing](#data-loading--preprocessing)
    - [Model Configuration](#model-configuration)
    - [Training and Inference](#training-and-inference)
5. [Module Details](#module-details)
    - [Data & Augmentation](#data--augmentation)
    - [Models](#models)
        - [APDTFlow with Learnable Time Series Embedding](#apdtflow-with-learnable-time-series-embedding)
        - [TransformerForecaster](#transformerforecaster)
        - [TCNForecaster](#tcnforecaster)
        - [EnsembleForecaster](#ensembleforecaster)
    - [Training & Inference Scripts](#training--inference-scripts)
    - [Utilities](#utilities)
6. [Examples and Tutorials](#examples-and-tutorials)
7. [License](#license)

---
*For a quick overview of the project, please see the [Project README](https://github.com/yotambraun/APDTFlow/tree/main).*
---

## 1. Introduction

APDTFlow is a modular forecasting framework for time series data that combines advanced techniques such as Neural Ordinary Differential Equations (Neural ODEs), Transformer-based architectures, and probabilistic modeling. Its design emphasizes flexibility and extensibility, enabling researchers and practitioners to:
- Experiment with multiple forecasting architectures.
- Easily customize and extend components for novel research ideas.
- Benefit from built-in logging, checkpointing, and TensorBoard support.

---

## 2. Architecture Overview

APDTFlow’s architecture is organized into several key components:

- **Data Handling & Augmentation:**  
  Functions and classes to load time series data from CSV files, apply transformations (e.g., jitter, scaling, time warping), and construct sliding-window datasets.

- **Model Suite:**  
  The framework includes several forecasting models:
  - **APDTFlow Model:** Now enhanced with a learnable time series embedding module that processes raw temporal inputs using gated residual networks. This embedding enriches the signal before multi-scale decomposition, neural ODE dynamics, probabilistic fusion, and a time-aware Transformer decoder.
  - **TransformerForecaster:** Leverages Transformer architecture to capture long-range dependencies.
  - **TCNForecaster:** Uses Temporal Convolutional Networks for fast and efficient forecasting.
  - **EnsembleForecaster:** Combines multiple models to improve robustness.

- **Training & Inference:**  
  Ready-to-use scripts support model training and inference via command-line arguments and YAML configuration files.

- **Utilities & Logging:**  
  Helper functions for checkpointing, logging, and evaluation streamline development and debugging.

---


## 3. Installation and Setup

### From PyPI

APDTFlow is available on [PyPI](https://pypi.org/project/apdtflow). Install it with:

```bash
pip install apdtflow
```

### From Source
Clone the repository and install in editable mode:

```bash
git clone https://github.com/yourusername/apdtflow_project.git
cd apdtflow_project
pip install -e .
```
Ensure that you have Python 3.7 or higher, and install the required dependencies listed in requirements.txt.

---

## 4. Getting Started
### Data Loading & Preprocessing
APDTFlow provides a **TimeSeriesWindowDataset** class to handle data ingestion. This class:
* Reads CSV files containing your time series data.
* Sorts and processes the date column.
* Constructs sliding windows using two parameters:
    - `T_in`: The number of past observations used for prediction.
    - `T_out`: The forecast horizon (number of future time steps).

Example usage:
```python
from apdtflow.data import TimeSeriesWindowDataset
dataset = TimeSeriesWindowDataset(
    csv_file="path/to/your/dataset.csv",
    date_col="DATE",
    value_col="VALUE",
    T_in=12,  # Input sequence length
    T_out=3   # Forecast horizon
)
```
### Model Configuration
The APDTFlow model is configurable via several key parameters:
* **num_scales:** APDTFlow employs a multi-scale decomposition technique to capture both global and local trends in the data. The num_scales parameter determines how many scales (or resolutions) the input signal will be decomposed into. A higher number of scales may allow the model to capture more complex temporal patterns, but it could also increase computational complexity.
* **filter_size:** This parameter is used in the convolutional component (or dynamic convolution) within the model’s decomposer module. It defines the size of the convolutional filter applied to the input signal, thereby affecting the receptive field. A larger filter size allows the model to consider a broader context in the time series but may smooth out finer details.
* **forecast_horizon:** This parameter is used within the model to indicate the number of future time steps that the decoder will produce. It should match T_out to ensure consistency between the training data and the model's output.
* **hidden_dim:** The size of the hidden state in the dynamics module and decoder. This parameter controls the capacity of the model to learn complex representations. Increasing hidden_dim may improve the model’s performance, but at the cost of additional computational resources and potential overfitting if not tuned properly.
* **use_embedding:** A boolean flag to enable the learnable time series embedding.
* **embed_dim:** Dimension of the learned time series embedding (typically set equal to hidden_dim).

Example configuration:
```python
from apdtflow.models.apdtflow import APDTFlow

model = APDTFlow(
    num_scales=3,
    input_channels=1,
    filter_size=5,
    hidden_dim=16,
    output_dim=1,
    forecast_horizon=3,
    use_embedding=True
)
```

Or via the YAML configuration file (apdtflow/config/config.yaml):
```yaml
model: "APDTFlow"
csv_file: "dataset_examples/Electric_Production.csv"
date_col: "DATE"
value_col: "IPG2211A2N"
T_in: 12
T_out: 3
num_scales: 3
filter_size: 5
hidden_dim: 16
output_dim: 1
forecast_horizon: 3
batch_size: 16
learning_rate: 0.001
num_epochs: 15
checkpoint_dir: "checkpoints"
use_embedding: true
embed_dim: 16
```

### Training and Inference
APDTFlow provides scripts for both training and inference.

#### Training Script
Use the training script with appropriate command-line arguments:
```bash
python experiments/scripts/train.py \
  --csv_file "path/to/dataset.csv" \
  --date_col "DATE" \
  --value_col "VALUE" \
  --T_in 12 \
  --T_out 3 \
  --model APDTFlow \
  --num_scales 3 \
  --filter_size 5 \
  --hidden_dim 16 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --num_epochs 15 \
  --checkpoint_dir "./checkpoints"
```

#### Inference Script
Run inference using a saved checkpoint:
```bash
python experiments/scripts/infer.py \
  --csv_file "path/to/dataset.csv" \
  --date_col "DATE" \
  --value_col "VALUE" \
  --T_in 12 \
  --T_out 3 \
  --model APDTFlow \
  --checkpoint_path "./checkpoints/APDTFlow_checkpoint.pt" \
  --batch_size 16
```

Below is an example forecast produced by APDTFlow:
![APDTFlow Forecast](../assets/images/forecast_adtflow.png)

---
## 5. Module Details
### Data & Augmentation

* **Data Module:** Provides functions to read, sort, and preprocess CSV time series data.
* **Augmentation Functions::** Includes jitter, scaling, and time_warp to artificially enlarge the training data and make the model more robust.

### Models

The framework includes multiple forecasting models. For a detailed explanation of each model's architecture, parameters, and use cases, please refer to the [Model Architectures](models.md) documentation.

- **APDTFlow:** Integrates multi-scale decomposition, neural ODEs, probabilistic fusion, and a Transformer-based decoder and now includes a learnable time series embedding.
The APDTFlow model has been enhanced with a learnable TimeSeriesEmbedding module. This module uses gated residual networks (GRNs) to transform:  
    - Raw (normalized) time indices,
    - A periodic component (learned instead of fixed sinusoidal functions), and
    - Optionally, calendar features.
  These embeddings are projected back to a single channel and then processed by the multi-scale decomposer, neural ODE dynamics, probabilistic fusion, and a time-aware Transformer decoder.
- **TransformerForecaster:** Leverages the Transformer architecture for capturing long-range dependencies.
- **TCNForecaster:** Uses Temporal Convolutional Networks for efficient forecasting.
- **EnsembleForecaster:** Combines multiple models to improve robustness.

### Training & Inference Scripts
* **Training Script (train.py):** Handles data loading, model initialization, training loops, logging, and checkpoint saving.
* **Inference Script (infer.py):** Loads a trained model, processes input data, and computes evaluation metrics such as MSE and MAE.

### Utilities
* **Logging:** Configured to output both to the console and a log file.
* **Checkpointing:** Functions for saving and loading model states.
* **Evaluation Metrics:** Scripts to calculate error metrics and validate model performance.

---
## 6. Examples:

Visit the [APDTFlow Examples](https://github.com/yotambraun/APDTFlow/blob/main/experiments/notebooks/tutorial.ipynb) directory for Jupyter Notebook tutorials that demonstrate:
* Exploratory Data Analysis (EDA)
* Model training and hyperparameter tuning
* Running inference and interpreting results


---
## 7. License
APDTFlow is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.