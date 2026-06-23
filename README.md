# Deep Learning for Climate Analytics  
## Multi-Channel Temperature Sequence Modeling & Generation

An advanced deep learning framework designed to model historical climate data, capture complex seasonal dependencies, and evaluate both predictive and generative structures.

This project implements:

- A Multi-Channel Sequence-to-Sequence (Seq2Seq) Forecasting Engine
- A Variational Autoencoder (VAE) for structural climate simulation
- End-to-end tensor pipelines for multivariate temporal learning

The system was developed using historical monthly temperature data from multiple Western Australia weather stations.

---

# Project Overview

Accurately modeling and simulating multi-variable temporal sequences is critical for:

- Climate analytics
- Risk management
- Scenario simulation
- Quantitative forecasting
- Environmental derivatives research

This project explores two complementary paradigms:

### 1. Deterministic Forecasting

A recurrent Seq2Seq forecasting architecture predicts future 12-month joint temperature trajectories using a 72-month historical lookback window.

### 2. Generative Climate Simulation

A Variational Autoencoder (VAE) learns a compressed latent representation of seasonal climate structures and generates realistic synthetic multi-year temperature sequences.

---

# System Architecture

```text
                                      ┌──► Seq2Seq Forecasting Model ──► 12-Month Joint Prediction
                                      │    (Input: 72x2 -> Output: 12x2)
[Raw Climate Tensors] ──► [Slicing Engine]
  (N, 84 Months, 2)                   │
                                      └──► VAE Generative Framework ──► Latent Sampling & Synthesis
                                           (Learns 84x2 distribution)
```

---

# Data Engineering Pipeline

## Tensor Geometry

The climate sequences are represented as:

```python
(N, 84, 2)
```

Where:

- Channel 0 → Mean Minimum Temperatures
- Channel 1 → Mean Maximum Temperatures

---

## Forecast Window Construction

The forecasting engine applies a sliding temporal window:

### Input Tensor

```python
X ∈ R^(N × 72 × 2)
```

- 72-month historical lookback window

### Forecast Target

```python
y ∈ R^(N × 12 × 2)
```

- 12-month prediction horizon

---

## Data Partitioning Strategy

The preprocessing pipeline preserves strict temporal integrity:

- Training data shuffled for variance robustness
- Validation and test sets kept chronologically ordered
- True out-of-sample drift preserved for evaluation realism

---

# Model Architectures

## 1. Multi-Channel Seq2Seq Forecast Engine

The forecasting framework is built using recurrent neural networks optimized for long-horizon temporal state tracking.

### Key Features

- Multi-channel sequence modeling
- Joint minimum/maximum temperature forecasting
- Encoder-decoder recurrent architecture
- Long-term seasonal dependency learning
- Multi-step forecasting across 12-month horizons

The model encodes historical climate dynamics into latent temporal states and decodes future trajectories across both temperature channels simultaneously.

---

## 2. Variational Autoencoder (VAE)

An unsupervised latent variable generative framework designed to learn structural climatological manifolds.

### Core Components

- Encoder network
- Latent Gaussian embedding space
- Probabilistic sampling layer
- Decoder reconstruction network

### Optimization Objective

The VAE is trained using variational inference:

:contentReference[oaicite:0]{index=0}

This regularization framework ensures:

- Continuous latent interpolation
- Stable generative structure
- Reduced variance explosion
- Smooth seasonal synthesis

---

# Analytical Insights & Performance

## Forecasting Performance (Seq2Seq)

The forecasting engine demonstrates strong stability across unseen validation and test datasets.

### Observed Capabilities

- Captures seasonal turning points
- Preserves cross-channel dependencies
- Maintains smooth multi-step forecasts
- Tracks long-term climate trends effectively

---

## Generative Simulation Fidelity (VAE)

The VAE successfully learns long-term climatological structures.

### Key Observations

#### Seasonal Synchronization

Synthetic sequences reproduce:

- Summer peaks
- Winter troughs
- Multi-year cyclical behaviour

#### Stable Latent Sampling

Generated climate trajectories remain:

- Smooth
- Structurally coherent
- Free from unrealistic spikes
- Statistically stable across long horizons

The probabilistic structure of the VAE produces clean, averaged climate simulations suitable for baseline environmental scenario generation.

---

# Tech Stack

## Deep Learning Frameworks

- TensorFlow 2.x
- Keras
- Custom subclassed layers

## Numerical Computing

- NumPy
- Pandas

## Visualization

- Matplotlib
- Dynamic multi-axis temporal overlays

## Infrastructure & Execution

- JupyterLab
- Linux HPC environments
- SLURM batch scheduling

---

# Repository Structure

```text
.
├── README.md
├── Vu_angela_proj2.ipynb
└── climate dataset files
```

---

# Running the Project

## Clone Repository

```bash
git clone https://github.com/your-username/deep-learning-climate-analytics.git
cd deep-learning-climate-analytics
```

---

## Environment Setup

```bash
conda activate /group/cits5017/cits5017-2025
```

---

## Launch Jupyter Notebook

```bash
jupyter-lab Vu_angela_proj2.ipynb
```

---

## Run on HPC Cluster

```bash
sbatch project2.slurm
```

---

# Core Competencies Demonstrated

## Multivariate Temporal Modeling

- Multi-channel sequence learning
- Long-horizon forecasting
- Joint covariance preservation

## Deep Generative Modeling

- Variational inference
- Latent manifold learning
- Structural sequence generation

## ML Systems Engineering

- Tensor pipeline construction
- Temporal data slicing frameworks
- Sequence validation methodologies

## HPC Workflow Integration

- SLURM batch orchestration
- Scalable training execution
- Research-to-production workflow adaptation

---

# Key Learning Outcomes

This project strengthened practical experience in:

- Sequence-to-sequence deep learning
- Recurrent neural network architectures
- Variational Autoencoders (VAEs)
- Multivariate climate analytics
- Temporal tensor engineering
- Generative probabilistic modeling
- HPC-integrated machine learning workflows
