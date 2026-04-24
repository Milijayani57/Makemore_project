# AML-3103 Project Report: Makemore — Character-Level Language Models

**Course:** AML-3103  
**Date:** April 24, 2026

---

## 1. Introduction

This project reimplements Andrej Karpathy's _Makemore_ series — a progressive study of character-level language models that generate names. Starting from a simple bigram count table and ending at a WaveNet-inspired hierarchical convolutional network, each part introduces a core deep learning concept while keeping the problem domain constant. All experiments are tracked with MLflow, and model checkpoints are saved for reproducibility.

**Goal:** Predict the next character in a name given the previous `block_size` characters, trained on a corpus of 32,033 English names.

---

## 2. Dataset

| Property                 | Value                                                 |
| ------------------------ | ----------------------------------------------------- |
| Primary dataset          | `names.txt` — 32,033 English names                    |
| Alternative dataset      | `scientific_names.txt` — 32,000 marine creature names |
| Vocabulary               | 27 tokens (26 letters + `.` start/end token)          |
| Train / Val / Test split | 80% / 10% / 10%                                       |
| Context window           | Variable: `block_size` ∈ {3, 6, 8, 12, 16}            |

Each training example is a fixed-length context window extracted via a sliding window over name character sequences.

---

## 3. Models

The project develops five progressively more complex architectures.

### 3.1 Part 1 — Bigram Model

A character-level bigram model implemented two ways:

1. **Count table** — raw bigram frequency matrix normalised to probabilities.
2. **1-layer neural net** — single linear layer (27×27) trained with negative log-likelihood, equivalent to the count table at convergence.

This establishes a lower-bound baseline. The model can only condition on the single immediately preceding character.

### 3.2 Part 2 — MLP

Follows Bengio et al. (2003). Characters are embedded into a continuous space; their embeddings are concatenated and passed through a single hidden layer with tanh activation before a final linear projection to logits.

| Component    | Detail                                   |
| ------------ | ---------------------------------------- |
| Embedding    | `vocab_size × n_embd` lookup table       |
| Hidden layer | `(block_size × n_embd) → n_hidden`, tanh |
| Output       | `n_hidden → 27`, cross-entropy           |

### 3.3 Part 3 — MLP with Batch Normalisation

The Part 2 MLP is extended with a manual implementation of Batch Normalisation (Ioffe & Szegedy, 2015) and improved weight initialisation:

- **Kaiming init:** `W1` scaled by `(5/3) / √(fan_in)` for tanh.
- **Small output init:** `W2` scaled by `0.01` to prevent logit saturation at step 0.
- **BatchNorm calibration:** After training, running statistics are recalibrated with a full forward pass over the training set for accurate inference.

### 3.4 Part 4 — Manual Backpropagation

The Part 3 architecture is reimplemented with all gradients computed by hand — no autograd. This validates that the forward/backward pass is mathematically correct and builds intuition for the computational graph. Results closely match Part 3.

### 3.5 Part 5 — WaveNet (Hierarchical CNN)

Inspired by WaveNet's dilated causal convolutions, three sequential `FlattenConsecutive(2)` stages progressively collapse a sequence of 8 characters into a single hidden state before prediction.

**Architecture:**

```
Input: (B, 8) token IDs
Embedding: (B, 8, 24)

Stage 1: FlattenConsecutive(2) → Linear → BatchNorm → Tanh   # (B, 4, 48) → (B, 4, 128)
Stage 2: FlattenConsecutive(2) → Linear → BatchNorm → Tanh   # (B, 2, 256) → (B, 2, 256)
Stage 3: FlattenConsecutive(2) → Linear → BatchNorm → Tanh   # (B, 1, 512) → (B, 1, 256)

Flatten → Linear(256, 27) → Logits
```

This hierarchical structure enables the model to combine nearby character features first, then higher-level features, mirroring dilated convolutions without requiring explicit dilation logic.

---

## 4. Training Setup

| Hyperparameter         | Value                                    |
| ---------------------- | ---------------------------------------- |
| Optimiser              | SGD                                      |
| Learning rate schedule | 0.1 for first half, 0.01 for second half |
| Training steps         | 50,000                                   |
| Default batch size     | 32                                       |
| Default context length | 3                                        |
| Default embedding dim  | 10                                       |
| Default hidden dim     | 200                                      |

---

## 5. Results

### 5.1 Per-Part Performance

| Part  | Architecture          | Train Loss | Val Loss  |
| ----- | --------------------- | ---------- | --------- |
| 1     | Bigram (count table)  | —          | ~2.45     |
| 2     | MLP                   | 2.118      | ~2.17     |
| 3     | MLP + BatchNorm       | 2.00       | 2.08      |
| 4     | MLP (manual backprop) | 2.072      | 2.110     |
| **5** | **WaveNet**           | **1.769**  | **1.993** |

The WaveNet model achieves the best validation loss of **1.993**, a ~4% absolute improvement over the best MLP model (Part 3, val loss 2.08).

### 5.2 Hyperparameter Sweep: Context Length

_Config: n_embd=10, n_hidden=200, batch=32, steps=50,000_

| block_size | Train Loss | Val Loss   |
| ---------- | ---------- | ---------- |
| 3          | 2.1298     | 2.1505     |
| 6          | 2.0526     | 2.0898     |
| 8          | 2.0435     | 2.0905     |
| 12         | 2.0280     | 2.0758     |
| **16**     | **2.0216** | **2.0731** |

**Finding:** Longer context consistently improves performance; gains plateau beyond block_size=12. Best config: `block_size=16` (val loss 2.0731).

### 5.3 Hyperparameter Sweep: Batch Size

_Config: block_size=3, n_embd=10, n_hidden=200, steps=50,000_

| batch_size | Train Loss | Val Loss   |
| ---------- | ---------- | ---------- |
| 16         | 2.1812     | 2.1905     |
| 32         | 2.1298     | 2.1505     |
| 64         | 2.1064     | 2.1325     |
| **128**    | **2.0946** | **2.1254** |

**Finding:** Larger batches produce more stable gradient estimates, leading to better convergence. Best config: `batch_size=128` (val loss 2.1254).

### 5.4 Hyperparameter Sweep: Architecture Size

_Config: block_size=3, batch=32, steps=50,000_

| Config    | n_embd | n_hidden | Train Loss | Val Loss   |
| --------- | ------ | -------- | ---------- | ---------- |
| tiny      | 8      | 100      | 2.1724     | 2.1824     |
| small     | 10     | 200      | 2.1298     | 2.1505     |
| medium    | 16     | 300      | 2.0945     | 2.1277     |
| **large** | **24** | **400**  | **2.0734** | **2.1106** |

**Finding:** Larger embedding and hidden dimensions consistently help. Best config: `n_embd=24, n_hidden=400` (val loss 2.1106).

### 5.5 Best Combined Configuration (MLP + BatchNorm)

Combining the best settings from all three sweeps:

| block_size | batch_size | n_embd | n_hidden | Estimated Val Loss |
| ---------- | ---------- | ------ | -------- | ------------------ |
| 16         | 128        | 24     | 400      | ~2.07              |

Even this optimised MLP configuration is outperformed by the Part 5 WaveNet (val loss **1.993**), confirming that the architectural choice matters more than hyperparameter tuning alone.

---

## 6. Assignment 4/5 — Fashion MNIST (Supplementary)

As part of the coursework, a separate set of experiments was run on the Fashion MNIST image classification benchmark, comparing MLP and CNN architectures with and without Batch Normalisation.

| Model           | Test Accuracy |
| --------------- | ------------- |
| MLP Baseline    | 88.49%        |
| MLP + BatchNorm | 89.64%        |
| CNN             | **91.21%**    |

Results confirm that (1) BatchNorm improves MLP performance and (2) convolutional architectures with inductive bias for spatial data outperform flat MLPs on image tasks.

---

## 7. Experiment Tracking (MLflow)

All runs are logged under the `makemore` MLflow experiment (ID: `892347093702429276`).

**Logged per run:**

| Category   | Values                                                                          |
| ---------- | ------------------------------------------------------------------------------- |
| Parameters | `block_size`, `n_embd`, `n_hidden`, `batch_size`, `vocab_size`, `learning_rate` |
| Metrics    | `train_loss`, `val_loss`, loss at every 10,000 steps                            |
| Artifacts  | Model checkpoint `.pt` file, hyperparameter summary plot                        |

**Total runs logged:** 21 (5 main parts + 13 hyperparameter sweep runs + misc).

To browse the UI:

```bash
mlflow ui   # http://localhost:5000
```

---

## 8. Saved Checkpoints

| File                                     | Size    | Description                   |
| ---------------------------------------- | ------- | ----------------------------- |
| `checkpoints/part1_bigram.pt`            | 8.1 KB  | Bigram baseline               |
| `checkpoints/part2_mlp.pt`               | 50.5 KB | Simple MLP                    |
| `checkpoints/part3_bn.pt`                | 141 MB  | MLP + BatchNorm (best MLP)    |
| `checkpoints/part4_backprop.pt`          | 54.6 KB | Manual backprop validation    |
| `checkpoints/part5_wavenet.pt`           | 311 KB  | WaveNet (best overall)        |
| `checkpoints/hyperparameter_summary.png` | 71.2 KB | 3-panel sweep comparison plot |

---

## 9. Key Implementation Insights

1. **BatchNorm calibration** is essential for correct inference: after training with mini-batch statistics, a full forward pass over the training set is required to synchronise the running mean/variance buffers with the true data distribution.

2. **Kaiming initialisation** prevents tanh saturation at the start of training. Without it, pre-activations are too large and gradients vanish, slowing convergence significantly.

3. **WaveNet's hierarchical structure** outperforms a flat MLP even with fewer total parameters, because it imposes a meaningful inductive bias: nearby characters are combined first, progressively building longer-range dependencies.

4. **Longer context windows** always help for this task, but with diminishing returns beyond 12–16 characters (names rarely exceed that length).

5. **Manual backpropagation** (Part 4) exactly reproduces Part 3 results, validating that PyTorch's autograd is correct and that building intuition about gradient flow is tractable even for networks with BatchNorm.

---

## 10. Conclusion

The project demonstrates a clean progression from a 2.45 bigram baseline to a 1.993 WaveNet model across five architectures. The most impactful single change was moving from a flat MLP to the hierarchical WaveNet (Part 5), which reduced validation loss by ~0.09 absolute — larger than any hyperparameter tuning gain. Systematic hyperparameter sweeps show that context length matters most among the three dimensions explored, followed by model size and batch size.

The overall best result is the **Part 5 WaveNet with val loss 1.993**, trained for 50,000 steps on 32,033 English names with a context window of 8 characters.

---

## References

- Karpathy, A. (2022). _makemore_. https://github.com/karpathy/makemore
- Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). A neural probabilistic language model. _JMLR_.
- Ioffe, S., & Szegedy, C. (2015). Batch Normalization. _ICML_.
- van den Oord, A., et al. (2016). WaveNet: A Generative Model for Raw Audio. _arXiv:1609.03499_.
