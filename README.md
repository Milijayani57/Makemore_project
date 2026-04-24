# Makemore — AML-3103 Semester Project

Character-level language model reimplementations following Andrej Karpathy's Makemore series, with MLflow experiment tracking and hyperparameter sweeps.

## Project Structure

```
.
├── names.txt                          # Training data (32,033 names)
├── scientific_names.txt               # Marine creature data (32,000 names)
├── project.md                         # Project brief
├── create_experiments_nb.py           # Generates experiments.ipynb via nbformat
├── experiments.ipynb                  # Hyperparameter sweep (13 runs)
│
├── refactored/                        # Main training notebooks (MLflow-instrumented)
│   ├── makemore_part1_bigrams_refactored.ipynb
│   ├── makemore_part2_mlp_refactored.ipynb
│   ├── makemore_part3_bn_refactored.ipynb
│   ├── makemore_part4_backprop_refactored.ipynb
│   └── makemore_part5_cnn1_refactored.ipynb
│
├── checkpoints/                       # Saved model weights (.pt) + summary plot
└── mlruns/                            # MLflow tracking data (auto-generated)
```

## Parts Overview

| Part | Concept | Architecture | Val Loss |
|------|---------|--------------|----------|
| 1 | Bigram language model | Count table + 1-layer neural net | ~2.45 |
| 2 | MLP | Embedding lookup + 1 hidden layer (tanh) | ~2.17 |
| 3 | Activations & BatchNorm | MLP + manual BatchNorm, Kaiming init | ~2.08 |
| 4 | Backprop | Part 3 architecture, gradients by hand | ~2.11 |
| 5 | WaveNet | Hierarchical `FlattenConsecutive×3`, block_size=8 | **1.993** |

## Requirements

```bash
pip install torch numpy matplotlib mlflow nbformat jupyter
```

## Reproducing Results

Run all five notebooks:

```bash
# Must exectue sequentially before starting others (sets up checkpoints/)
jupyter nbconvert --to notebook --execute refactored/makemore_part1_bigrams_refactored.ipynb
  


# Hyperparameter experiments (context length, batch size, architecture)
python create_experiments_nb.py
jupyter nbconvert --to notebook --execute experiments.ipynb
```

View all runs in MLflow:

```bash
mlflow ui   # http://localhost:5000
```

## Hyperparameter Sweep Results

All sweeps use the Part 3 (MLP + BatchNorm) architecture as the baseline, 50,000 training steps each.

### Context Length (n_embd=10, n_hidden=200, batch=32)

| block_size | train | val |
|------------|-------|-----|
| 3 | 2.1298 | 2.1505 |
| 6 | 2.0526 | 2.0898 |
| 8 | 2.0435 | 2.0905 |
| 12 | 2.0280 | 2.0758 |
| **16** | **2.0216** | **2.0731** |

### Batch Size (block=3, n_embd=10, n_hidden=200)

| batch_size | train | val |
|------------|-------|-----|
| 16 | 2.1812 | 2.1905 |
| 32 | 2.1298 | 2.1505 |
| 64 | 2.1064 | 2.1325 |
| **128** | **2.0946** | **2.1254** |

### Architecture Size (block=3, batch=32)

| config | n_embd | n_hidden | train | val |
|--------|--------|----------|-------|-----|
| tiny | 8 | 100 | 2.1724 | 2.1824 |
| small | 10 | 200 | 2.1298 | 2.1505 |
| medium | 16 | 300 | 2.0945 | 2.1277 |
| **large** | **24** | **400** | **2.0734** | **2.1106** |

Best standalone config: block_size=16, batch=128, n_embd=24, n_hidden=400 → val ≈ 2.07.
Best overall: Part 5 WaveNet with hierarchical convolutions → **val = 1.993**.

## MLflow Experiment Tracking

Every run logs:
- **Params**: block_size, embed_dim, hidden_dim, batch_size, vocab_size, learning_rate
- **Metrics**: train_loss, val_loss (+ per-10k-step loss curves in experiment runs)
- **Artifacts**: model checkpoint `.pt` file

All runs are grouped under the `makemore` experiment. Model weights are saved as PyTorch state dicts in `checkpoints/`.

## Key Implementation Notes

- **BatchNorm calibration**: After training, a full forward pass over the training set recalibrates running mean/variance for accurate inference — critical for Parts 3 and 4.
- **Kaiming initialization**: `W1` scaled by `(5/3) / sqrt(fan_in)` for tanh nonlinearities; `W2` initialized small (`× 0.01`) to avoid logit saturation at step 0.
- **WaveNet hierarchy**: Three `FlattenConsecutive(2)` stages collapse a sequence of 8 characters into a single hidden state, matching the dilated convolution intuition from WaveNet.
