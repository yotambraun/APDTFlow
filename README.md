# APDTFlow: A Modular Forecasting Framework for Time Series Data

[![PyPI version](https://img.shields.io/pypi/v/apdtflow.svg)](https://pypi.org/project/apdtflow)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

APDTFlow is a modern and extensible forecasting framework for time series data that leverages advanced techniques including neural ordinary differential equations (Neural ODEs), transformer-based components, and probabilistic modeling. Its modular design allows researchers and practitioners to experiment with multiple forecasting models and easily extend the framework for new methods.

## Features

- **Differentiable Programming with Neural ODEs:** Capture complex temporal dynamics with a hierarchical dynamics module.
- **Transformer-based Decoding:** Use a time-aware Transformer decoder to effectively model time dependencies.
- **Probabilistic Modeling:** Quantify uncertainty with variational updates and probabilistic fusion techniques.
- **Modular Design:** Easily swap out or extend components (e.g. decomposers, decoders, dynamics) for custom forecasting solutions.
- **Built-in Logging and TensorBoard Support:** Monitor training progress with professional logging and optional TensorBoard integration.
- **Multiple Architectures:** Choose from APDTFlow, Transformer‑ and TCN‑based forecasters, or build ensembles.

## Installation

APDTFlow is published on [PyPI](https://pypi.org/project/apdtflow). Install it using `pip`:

```bash
pip install apdtflow
```

Alternatively, to install from source in editable mode (for development):

```bash
git clone https://github.com/yourusername/apdtflow_project.git
cd apdtflow_project
pip install -e .
```

## Quick Start
Here is a brief example showing how to use APDTFlow for training and inference:

### Training

```python
import torch
from torch.utils.data import DataLoader
from apdtflow.data import TimeSeriesWindowDataset
from apdtflow.models.apdtflow import APDTFlow

# Create dataset
dataset = TimeSeriesWindowDataset(
    csv_file="path/to/your/dataset.csv",
    date_col="DATE",
    value_col="VALUE",
    T_in=12,
    T_out=3
)

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize the model
model = APDTFlow(
    num_scales=3,
    input_channels=1,
    filter_size=5,
    hidden_dim=16,
    output_dim=1,
    forecast_horizon=3
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train the model
model.train_model(
    train_loader=train_loader,
    num_epochs=15,
    learning_rate=0.001,
    device=device
)
```

### Inference

```python
import torch
from torch.utils.data import DataLoader
from apdtflow.data import TimeSeriesWindowDataset
from apdtflow.models.apdtflow import APDTFlow

# Create test dataset
test_dataset = TimeSeriesWindowDataset(
    csv_file="path/to/your/dataset.csv",
    date_col="DATE",
    value_col="VALUE",
    T_in=12,
    T_out=3
)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Initialize the model and load the checkpoint
model = APDTFlow(
    num_scales=3,
    input_channels=1,
    filter_size=5,
    hidden_dim=16,
    output_dim=1,
    forecast_horizon=3
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

checkpoint_path = "path/to/your/checkpoint.pt"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))

# Evaluate the model
mse, mae = model.evaluate(test_loader, device)
print(f"Test MSE: {mse:.4f}, Test MAE: {mae:.4f}")
```

## Running Training and Inference from the Command Line
Your commands already show how to run the training and inference scripts. For clarity, here’s an explanation of the commands you provided:
### Training
Run the training script with your dataset and hyperparameters:

```bash
python experiments/scripts/train.py \
  --csv_file "C:/Users/yotam/code_projects/APDTFlow/dataset_examples/Electric_Production.csv" \
  --date_col "DATE" \
  --value_col "IPG2211A2N" \
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
This command does the following:

* --csv_file: Specifies the path to your dataset.
* --date_col and --value_col: Define the column names for dates and values.
* --T_in and --T_out: Set the input sequence length and forecast horizon.
* --model: Chooses the forecasting model (in this case, APDTFlow).
* Other arguments: Set additional model parameters and training configurations.
* --checkpoint_dir: Determines where to save the checkpoint after training.

### Inference
Run the inference script using the trained checkpoint:

```bash
python experiments/scripts/infer.py \
  --csv_file "C:/Users/yotam/code_projects/APDTFlow/dataset_examples/Electric_Production.csv" \
  --date_col "DATE" \
  --value_col "IPG2211A2N" \
  --T_in 12 \
  --T_out 3 \
  --model APDTFlow \
  --checkpoint_path "./checkpoints/APDTFlow_checkpoint.pt" \
  --batch_size 16
```
This command:

* Loads the dataset using the specified CSV file.
* Loads the trained model from the checkpoint file (APDTFlow_checkpoint.pt).
* Evaluates the model using the test data, printing metrics (e.g., MSE and MAE).

These examples can be demonstrated during a presentation or written in your documentation to show users how to work with the framework from the command line.

## Documentation
Full documentation is available in the [APDTFlow documentation](docs/index.md).

## Configuration
APDTFlow supports configuration via YAML files. A sample configuration file is available at [apdtflow/config/sample_config.yaml](apdtflow/config/sample_config.yaml).
 You can override these settings using command-line arguments when running the provided scripts (see the experiments/scripts/ directory).

## License
APDTFlow is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
