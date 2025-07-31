# Noise-efficient Private Dataset Distillation

This repository provides an implementation of a framework for noise-efficient differentially private dataset distillation. The code is adapted from [Differentially Private Dataset Condensation](https://openreview.net/forum?id=H8XpqEkbua_).

## Overview

The framework introduces **Decoupled Optimization and Sampling (DOS)** and **Subspace discovery for Error Reduction (SER)** to improve the utility of distilled datasets under differential privacy constraints.

## Prerequisites

See requirements.txt

## Usage

### Step 1: Compute Noise Magnitude
To calculate the noise scale (`sigma`) for a given privacy budget:

```bash
python compute_sigma_with_fixed_budget.py
```

Specify your privacy budget and dataset parameters in the script.

### Step 2: Dataset Distillation
Using the computed `sigma`, run the distillation process:

```bash
CUDA_VISIBLE_DEVICES=1 python dosser.py \
    --sampling_iteration 1000 \
    --training_iteration 200000 \
    --dataset CIFAR10 \
    --aux_path /data/rzheng/sd_cifar10_50000_96 \
    --aux_ipc 100 \
    --ser_dim 1000 \
    --SER --PEA
```

### Parameters:
- `--sampling_iteration`: Number of sampling iterations.
- `--training_iteration`: Number of optimization iterations.
- `--dataset`: Dataset to be used (e.g., CIFAR10).
- `--aux_path`: Path to auxiliary dataset.
- `--aux_ipc`: Images per class for auxiliary dataset.
- `--ser_dim`: Dimension of the subspace for SER.
- `--SER`: Enable Subspace Error Reduction (SER).
- `--PEA`: Use Partitioning and Expansion Augmentation (optional)