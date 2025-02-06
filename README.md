# APDTFlow: A Modular Forecasting Framework for Time Series Data
[![PyPI version](https://img.shields.io/pypi/v/apdtflow.svg)](https://pypi.org/project/apdtflow)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

APDTFlow is a modern and extensible forecasting framework for time series data that leverages advanced techniques including neural ordinary differential equations (Neural ODEs), transformer-based components, and probabilistic modeling. Its modular design allows researchers and practitioners to experiment with multiple forecasting models and easily extend the framework for new methods.

![APDTFlow Forecast](assets/images/forecast_adtflow.png)

## Features

- **Differentiable Programming with Neural ODEs:** Capture complex temporal dynamics with a hierarchical dynamics module.
- **Transformer-based Decoding:** Use a time-aware Transformer decoder to effectively model time dependencies.
- **Probabilistic Modeling:** Quantify uncertainty with variational updates and probabilistic fusion techniques.
- **Modular Design:** Easily swap out or extend components (e.g. decomposers, decoders, dynamics) for custom forecasting solutions.
- **Built-in Logging and TensorBoard Support:** Monitor training progress with professional logging and optional TensorBoard integration.
- **Multiple Architectures:** Choose from APDTFlow, Transformer‑ and TCN‑based forecasters, or build ensembles.

## Experiment Results
In our mega experiment we compared multiple forecasting models across different forecast horizons using 3-fold cross‑validation. For brevity, below we show two key plots:

1. **Validation Loss Comparison:** A bar plot comparing the average validation losses of the models (APDTFlow, TransformerForecaster, TCNForecaster, and EnsembleForecaster) across forecast horizons.
2. **Example Forecast (Horizon 7, CV Split 3):** A forecast plot for the APDTFlow model for a 7-step forecast from CV split 3.

## 1. Validation Loss Comparison

The bar plot below summarizes the average validation losses (lower is better) for the different models across the forecast horizons (7, 10, and 30 time steps):

![Validation Loss Comparison](experiments/results_plots/Validation_Loss_Comparison.png)

*Explanation:*  
This plot shows that the APDTFlow model (and possibly the ensemble) generally achieved lower validation losses compared to the other models, especially for longer forecast horizons. This indicates that its multi-scale decomposition and neural ODE dynamics are well-suited for capturing the trends and seasonal patterns in the dataset.
*Discussion:*  
The plot demonstrates that, overall, the APDTFlow model (and, in some cases, the ensemble) tend to achieve lower validation losses—particularly as the forecast horizon increases.

## 2. Performance vs. Forecast Horizon

The following line plot illustrates how the performance (average validation loss) of each model changes with different forecast horizons. This visualization helps to assess which models maintain consistent performance as the forecast horizon increases.

![Performance vs. Horizon](experiments/results_plots/Performance_vs_Horizon.png)

*Discussion:*  
The line plot reveals the trend in model performance across forecast horizons. It helps us understand which models degrade gracefully (or even improve) as the forecast horizon lengthens.

## 3. Example Forecast (Horizon 7, CV Split 3)

Below is an example forecast produced by the APDTFlow model for a forecast horizon of 7 time steps on the third cross-validation split.

![APDTFlow Forecast Horizon 7, CV3](experiments/results_plots/APDTFlow_Forecast_Horizon_7_CV3.png)

*Discussion:*  
- **Input Sequence (Blue):** The historical data (last 30 time steps) used as input.
- **True Future (Dashed Orange):** The actual future values for the next 7 time steps.
- **Predicted Future (Dotted Line):** The forecast generated by the model.

---

*For a detailed explanation, more plots, and additional analysis of these results, please see our [Experiment Results and Analysis](docs/experiment_results.md) document.*

---


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
    T_in=12,    # Input sequence length
    T_out=3     # Forecast horizon (number of future time steps)
)

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize the model
model = APDTFlow(
    num_scales=3,         # Number of scales to decompose the input signal
    input_channels=1,     # Number of input channels (typically 1 for univariate time series)
    filter_size=5,        # Filter size for the dynamic convolution (affects receptive field)
    hidden_dim=16,        # Hidden dimension size for the dynamics module and decoder
    output_dim=1,         # Output channels (typically 1 for univariate forecasts)
    forecast_horizon=3    # Number of future time steps to forecast (should match T_out)
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

## Core Model Parameters Explained:
For a comprehensive description of each model's architecture and additional details, please see the [Model Architectures Documentation](docs/models.md).
When configuring APDTFlow, several parameters play key roles in how the model processes and forecasts time series data. Here’s what they mean:

* **T_in (Input Sequence Length):** This parameter specifies the number of past time steps the model will use as input. For example, if T_in=12, the model will use the previous 12 observations to make a forecast.
* **T_out (Forecast Horizon):** This parameter defines the number of future time steps to predict. For instance, if T_out=3, the model will output predictions for the next 3 time steps.
* **num_scales:** APDTFlow employs a multi-scale decomposition technique to capture both global and local trends in the data. The num_scales parameter determines how many scales (or resolutions) the input signal will be decomposed into. A higher number of scales may allow the model to capture more complex temporal patterns, but it could also increase computational complexity.
* **filter_size:** This parameter is used in the convolutional component (or dynamic convolution) within the model’s decomposer module. It defines the size of the convolutional filter applied to the input signal, thereby affecting the receptive field. A larger filter size allows the model to consider a broader context in the time series but may smooth out finer details.
* **forecast_horizon:** This parameter is used within the model to indicate the number of future time steps that the decoder will produce. It should match T_out to ensure consistency between the training data and the model's output.
* **hidden_dim:** The size of the hidden state in the dynamics module and decoder. This parameter controls the capacity of the model to learn complex representations. Increasing hidden_dim may improve the model’s performance, but at the cost of additional computational resources and potential overfitting if not tuned properly.


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
APDTFlow supports configuration via YAML files. A sample configuration file is available at [config.yaml](https://github.com/yotambraun/APDTFlow/blob/main/apdtflow/config/config.yaml).
 You can override these settings using command-line arguments when running the provided scripts (see the experiments/scripts/ directory).

## License
APDTFlow is licensed under the MIT License. See the [LICENSE](License) file for more details.
