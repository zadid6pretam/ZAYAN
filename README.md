# ZAYAN: Disentangled Contrastive Transformer for Tabular Remote Sensing Data

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)
![PyPI](https://img.shields.io/badge/PyPI-zayan-blue.svg)
![Status](https://img.shields.io/badge/status-ICPR%202026%20accepted-purple.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

**ZAYAN** is a self-supervised, feature-centric contrastive learning framework for tabular remote sensing and environmental data. It learns robust, redundancy-minimized feature embeddings using feature-level contrastive pretraining and then uses those embeddings inside a Transformer classifier for downstream prediction.

The paper **“ZAYAN: Disentangled Contrastive Transformer for Tabular Remote Sensing Data”** has been accepted for presentation at the **28th International Conference on Pattern Recognition (ICPR 2026)** in Lyon, France.

---

## Architecture

<p align="center">
  <img src="ZAYAN_Architecture.png" alt="ZAYAN architecture" width="100%">
</p>

<p align="center">
  <em>
  Overview of ZAYAN: tabular features are augmented using noise, warping, and masking; encoded through the ZAYAN-CL feature-level contrastive module; regularized with a redundancy penalty; and then passed to the ZAYAN-T Transformer classifier for final prediction.
  </em>
</p>

---

## Overview

ZAYAN stands for **Zero-Anchor dYnamic feAture eNcoding**. It consists of two main modules:

1. **ZAYAN-CL**  
   A feature-level contrastive learning module that learns feature embeddings without class labels or explicit sample anchors.

2. **ZAYAN-T**  
   A Transformer-based classifier that uses the pretrained feature embeddings as structured feature tokens for supervised classification.

3. **ZAYAN**  
   A high-level wrapper that runs contrastive pretraining, supervised Transformer training, and evaluation.

Unlike conventional sample-level contrastive learning, ZAYAN performs contrastive learning at the **feature level**. It aligns augmented views of the same feature while reducing redundancy among different feature embeddings.

---

## Key Features

- Feature-level self-supervised contrastive pretraining
- Zero-anchor contrastive objective for tabular remote sensing data
- Redundancy reduction through Gram-matrix decorrelation
- Transformer classifier conditioned on learned feature embeddings
- Preservation loss to retain the geometry learned during contrastive pretraining
- Support for multiclass and binary classification
- Class-balanced supervised training
- Evaluation with accuracy, precision, recall, and F1-score
- Demo notebook with Optuna-based hyperparameter tuning
- Urban Land Cover experiment with diagnostics and analysis

---

## Repository Structure

```text
.
├── assets/
│   └── ZAYAN_Architecture.png
├── zayan/
│   ├── __init__.py
│   └── zayan.py
├── notebooks/
│   └── ZAYAN_Experiment.ipynb
├── README.md
├── pyproject.toml
├── LICENSE
└── MANIFEST.in
