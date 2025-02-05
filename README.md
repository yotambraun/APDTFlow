# APDTFlow Forecasting Framework

APDTFlow is a modern, modular, and flexible forecasting framework for time series data. It integrates advanced techniques such as:
- **Differentiable Programming with Neural ODEs:** A hierarchical dynamics module that captures both global and local time dependencies.
- **Transformer-based Components:** A time-aware Transformer decoder with sine–cosine positional encodings for capturing temporal patterns.
- **Probabilistic Modeling:** Variational updates and probabilistic fusion to quantify uncertainty.
- **Multiple Architectures & Ensemble Options:** In addition to the APDTFlow model, the framework provides Transformer‑ and TCN‑based forecasters, and supports building ensembles.

## Features

- **Modular Design:** Easily swap out components (dynamics module, decoder, etc.) or add new forecasting models.
- **Configuration Driven:** Use YAML configuration files or command‑line arguments to specify hyperparameters. A sample config file is provided at `apdtflow/config/sample_config.yaml`.
- **Professional Logging:** Integrated logging (via Python’s logging module) writes timestamped logs to both the console and files.
- **TensorBoard Support:** Optionally log training progress to TensorBoard.
- **Evaluation Metrics:** Compute MSE, MAE, and (optionally) additional metrics.
- **Extensible & Research-Ready:** Suitable for academic research and production environments.

## Installation

1. **Clone the Repository:**

```bash
   git clone https://github.com/yourusername/apdtflow_project.git
   cd apdtflow_project
```
2. **Create and Activate a Virtual Environment:**
On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```
On Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install in Editable Mode:**

```bash
pip install -e .
```
This installs APDTFlow in development mode, so changes are immediately reflected.

## Configuration
A sample configuration file is provided at apdtflow/config/sample_config.yaml:
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
batch_size: 16
learning_rate: 0.001
num_epochs: 15
checkpoint_dir: "checkpoints"
```
Usage Options:
- **Configuration File::** Supply a YAML configuration file via the --config command-line argument.
- **Command‑line Arguments:** Override any settings directly when running scripts.

## Usage
### Training
Using a configuration file:
```bash
python experiments/scripts/train.py --config "apdtflow/config/sample_config.yaml" --tensorboard
```
Using command‑line arguments:
```bash
python experiments/scripts/train.py \
    --csv_file "dataset_examples/Electric_Production.csv" \
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
    --checkpoint_dir "checkpoints" \
    --tensorboard
```
During training, logs are written to logs/training.log and (if enabled) TensorBoard logs are saved under runs/experiment1.

### Inference
Run the inference script after training:
```bash
python experiments/scripts/infer.py \
    --csv_file "dataset_examples/Electric_Production.csv" \
    --date_col "DATE" \
    --value_col "IPG2211A2N" \
    --T_in 12 \
    --T_out 3 \
    --model APDTFlow \
    --checkpoint_path "checkpoints/APDTFlow_checkpoint.pt" \
    --batch_size 16
```
Inference logs are written to logs/inference.log and evaluation metrics (MSE and MAE) are printed to the terminal.