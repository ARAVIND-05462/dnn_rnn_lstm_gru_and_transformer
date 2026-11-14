# dnn_rnn_lstm_gru_and_transformer

## Deep Neural Networks — Assignment 2

## Project Overview

This repository contains two notebooks implementing recurrent neural networks and Transformer architectures.

| Notebook | Contents |
|---|---|
| `problem1.ipynb` | RNN, LSTM and GRU cells, HAR classification, synthetic adding task, gradient-flow analysis, hidden-size experiments and ablations |
| `problem2.ipynb` | Scaled dot-product attention, multi-head attention, Transformer encoder, HAR classification, Tiny Shakespeare language model and attention ablations |

## Experimental Configuration

- Random seeds: `42`, `123`, `2026`
- Problem 1 hidden size: `64`
- Problem 2 model width: `128`
- Problem 2 Task A: `8` attention heads
- Problem 2 Task B: `4` attention heads and `4` Transformer layers
- HAR sequence length: `128`
- HAR input channels: `9`
- Device: CPU or CUDA, selected automatically

## Datasets

The notebooks download the datasets automatically when required.

**1. UCI Human Activity Recognition (HAR)**

https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

Used for six-class human activity classification using smartphone inertial sensor data.

**2. Tiny Shakespeare**

https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

Used for character-level next-character language modelling.

Internet access is required for the initial download.

## Environment and Dependencies

The recorded development environment:

- Python 3.9
- PyTorch 2.2.2
- NumPy 1.26.4
- Pandas 2.1.3
- Scikit-learn 0.24.2

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Running the Notebooks

### Google Colab

1. Upload or open `problem1.ipynb` in Google Colab.
2. Select a GPU runtime if available.
3. Execute the notebook from top to bottom.
4. Repeat the process for `problem2.ipynb`.
5. Allow the notebooks to download the required datasets.
6. Collect the generated figures and experiment results.

### Local Execution

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter lab
```

Open each notebook and execute all cells sequentially.

## Generated Outputs

The notebooks generate:

- HAR class-distribution plots
- Confusion matrices
- Training and validation learning curves
- Recurrent gradient-flow visualizations
- Hidden-size ablation plots
- Attention heatmaps
- Attention-head ablation results
- Causal-mask comparison plots
- Context-length runtime measurements
- `results.csv` experiment summary

Typical output directories:

```text
data/
problem1/figures/
problem2/figures/
results.csv
```

## Reproducibility

- Three random seeds are used for repeated experiments.
- HAR validation data is separated by subject.
- Normalization statistics are calculated using training data.
- The official HAR test subjects remain separate from training and validation.
- The Tiny Shakespeare corpus is split chronologically.
- Causal masking is used for autoregressive language modelling to prevent access to future tokens.
- CUDA memory measurements require a CUDA-enabled runtime. They are not available when executing on CPU.


Generated datasets, figures and experiment summaries are created during notebook execution.

## References

1. Anguita, D. et al. “A Public Domain Dataset for Human Activity Recognition Using Smartphones.” ESANN, 2013.
2. Vaswani, A. et al. “Attention Is All You Need.” NeurIPS, 2017.
3. PyTorch Documentation: https://pytorch.org/docs/stable/
4. UCI Machine Learning Repository: https://archive.ics.uci.edu/
