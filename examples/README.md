# APDTFlow Examples

This directory contains ready-to-run examples demonstrating APDTFlow's capabilities.

## 🚀 Quick Start Examples

### 1. **[quickstart_easy_api.py](quickstart_easy_api.py)** - NEW!
The simplest way to get started with time series forecasting using APDTFlow's easy high-level API.

**What you'll learn:**
- How to use the simple `fit()`/`predict()` interface
- Making forecasts with uncertainty estimates
- Visualizing forecasts with built-in plotting
- Trying different model architectures (ODE, Transformer, TCN, Ensemble)

**Run it:**
```bash
cd examples
python quickstart_easy_api.py
```

**Perfect for:** Beginners and anyone who wants fast, easy forecasting

---

## 📓 Jupyter Notebook Examples

Located in `experiments/notebooks/`:

### 2. **[tutorial.ipynb](../experiments/notebooks/tutorial.ipynb)**
Comprehensive tutorial covering all major features of APDTFlow.

**What you'll learn:**
- Setting up datasets
- Training all model types (APDTFlow, Transformer, TCN, Ensemble)
- Evaluating model performance
- Custom training loops

**Perfect for:** Understanding the full API and advanced features

### 3. **[embedding_integ.ipynb](../experiments/notebooks/embedding_integ.ipynb)**
Demonstrates the learnable time series embedding feature.

**What you'll learn:**
- How to enable time series embeddings
- Benefits of learned temporal representations
- Comparing performance with/without embeddings

**Perfect for:** Advanced users seeking state-of-the-art performance

### 4. **[cross_val.ipynb](../experiments/notebooks/cross_val.ipynb) & [Cross_Validation.ipynb](../experiments/notebooks/Cross_Validation.ipynb)**
Time series cross-validation strategies.

**What you'll learn:**
- Rolling window validation
- Expanding window validation
- Blocked cross-validation
- Proper time series evaluation

**Perfect for:** Rigorous model evaluation

### 5. **[mega_experiment.ipynb](../experiments/notebooks/mega_experiment.ipynb)**
Large-scale experiment comparing all models across multiple forecast horizons.

**What you'll learn:**
- Systematic model comparison
- Performance across different forecast horizons
- Statistical analysis of results

**Perfect for:** Research and benchmarking

### 6. **[preprocessing.ipynb](../experiments/notebooks/preprocessing.ipynb)**
Data preprocessing and augmentation techniques.

**What you'll learn:**
- Handling missing values
- Data normalization
- Time series augmentation
- Feature engineering

**Perfect for:** Data preparation best practices

### 7. **[evaluation.ipynb](../experiments/notebooks/evaluation.ipynb)**
Model evaluation and metrics.

**What you'll learn:**
- Different evaluation metrics (MSE, MAE, RMSE, MAPE)
- Custom metric creation
- Comparing model performance

**Perfect for:** Understanding model quality

### 8. **[exogenous_variables.ipynb](../experiments/notebooks/exogenous_variables.ipynb)** - NEW in v0.2.0! 🚀
Using external features to boost accuracy by 30-50%.

**What you'll learn:**
- How to use exogenous variables (temperature, holidays, promotions)
- Past-observed vs future-known covariates
- Comparing fusion strategies (concat, gated, attention)
- Real-world examples with visualizations

**Perfect for:** Maximizing forecast accuracy with external data

### 9. **[conformal_prediction.ipynb](../experiments/notebooks/conformal_prediction.ipynb)** - NEW in v0.2.0! 📊
Rigorous uncertainty quantification with coverage guarantees.

**What you'll learn:**
- Split conformal prediction (simple & reliable)
- Adaptive conformal prediction (for non-stationary data)
- Guaranteed coverage intervals (e.g., 95%)
- When and why to use conformal prediction

**Perfect for:** Decision-making, risk management, safety-critical applications

---

## 🎯 Choose Your Path

### I want to start forecasting ASAP
→ Start with **quickstart_easy_api.py**

### I want to understand everything
→ Work through **tutorial.ipynb**

### I want maximum accuracy
→ Check out **embedding_integ.ipynb** and **mega_experiment.ipynb**

### I want robust validation
→ Explore **cross_val.ipynb**

### I have messy data
→ Learn from **preprocessing.ipynb**

### I want to use external features (NEW!)
→ Explore **exogenous_variables.ipynb** 🚀

### I need rigorous uncertainty quantification (NEW!)
→ Check out **conformal_prediction.ipynb** 📊

---

## 📊 Example Datasets

All examples use datasets from `dataset_examples/`:
- **Electric_Production.csv** - U.S. electric production monthly data
- **daily-minimum-temperatures-in-me.csv** - Daily temperature data
- **monthly-beer-production-in-austr.csv** - Monthly beer production
- **sales-of-shampoo-over-a-three-ye.csv** - Retail sales data

---

## 🆘 Need Help?

- 📖 **Documentation**: [docs/index.md](../docs/index.md)
- 🏗️ **Model Details**: [docs/models.md](../docs/models.md)
- 📈 **Experiment Results**: [docs/experiment_results.md](../docs/experiment_results.md)
- 🐛 **Issues**: [GitHub Issues](https://github.com/yotambraun/APDTFlow/issues)

---

## 🚀 Coming Soon

- Google Colab notebooks (click to run in browser!)
- Attention visualization examples
- Hyperparameter tuning with Optuna
- Pre-trained models and transfer learning
- MLOps integration (MLflow, Weights & Biases)

**Stay tuned for v0.3.0!**
