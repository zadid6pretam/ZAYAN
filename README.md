# ZAYAN: Disentangled Contrastive Transformer for Tabular Remote Sensing Data

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Task](https://img.shields.io/badge/Task-Tabular%20Remote%20Sensing-orange)
![Model](https://img.shields.io/badge/Model-ZAYAN-blueviolet)
![Architecture](https://img.shields.io/badge/Architecture-Disentangled%20Contrastive%20Transformer-informational)
![Learning](https://img.shields.io/badge/Learning-Feature--Level%20Contrastive-purple)
![Objective](https://img.shields.io/badge/Objective-Zero--Anchor%20Encoding-critical)
![Domain](https://img.shields.io/badge/Domain-Remote%20Sensing%20%2F%20Environmental%20Data-teal)
![Conference](https://img.shields.io/badge/Conference-ICPR%202026-blue)
![Status](https://img.shields.io/badge/Status-Accepted-brightgreen)
![PyPI](https://img.shields.io/badge/PyPI-zayan-blue)

<p align="center">
  <img src="ZAYAN_Architecture.png" alt="ZAYAN Architecture" width="1000">
</p>

ZAYAN is a self-supervised, feature-centric contrastive learning framework for **tabular remote sensing and environmental data**. ZAYAN stands for **Zero-Anchor dYnamic feAture eNcoding**. Rather than applying contrastive learning at the sample or image-patch level, it learns representations at the **feature level**, where each feature is dynamically augmented, encoded, contrasted, and regularized to reduce redundancy. The learned feature embeddings are then used by a Transformer classifier that preserves the contrastive feature geometry for downstream prediction. This design makes ZAYAN especially suitable for heterogeneous tabular sensing data derived from satellite products, GIS layers, environmental indicators, and remote-sensing-driven prediction tasks. Across multiple remote-sensing and environmental tabular benchmarks, ZAYAN achieves strong classification performance, robustness, and generalization compared with classical machine learning, tree ensembles, tabular neural networks, and recent tabular foundation-style baselines.

## Citation

Al Zadid Sultan Bin Habib, Tanpia Tasnim, Md. Ekramul Islam, and Muntasir Tabasum. **“ZAYAN: Disentangled Contrastive Transformer for Tabular Remote Sensing Data.”** In *Proceedings of the 28th International Conference on Pattern Recognition (ICPR)*, Lyon, France, 2026.

BibTeX:
```bibtex
@inproceedings{habib2026zayan,
  title     = {ZAYAN: Disentangled Contrastive Transformer for Tabular Remote Sensing Data},
  author    = {Habib, Al Zadid Sultan Bin and Tasnim, Tanpia and Islam, Md. Ekramul and Tabasum, Muntasir},
  booktitle = {Proceedings of the 28th International Conference on Pattern Recognition},
  year      = {2026},
  address   = {Lyon, France}
}

## Files and Repository Structure

### Python package: `zayan/`

This folder contains the core ZAYAN implementation:

- `__init__.py` - Package initializer and high-level API exports.
- `zayan.py` - Main ZAYAN implementation, including `ZAYAN_CL`, `ZAYAN_T`, and the high-level `ZAYAN` wrapper for contrastive pretraining, Transformer-based supervised training, and evaluation.

### Notebooks

- **`ZAYAN_Experiment.ipynb`**  
  Contains the main experiment notebook for ZAYAN. The notebook includes an Optuna-tuned run on the Urban Land Cover dataset, along with data preprocessing, ZAYAN-CL feature-level contrastive pretraining, ZAYAN-T supervised Transformer training, evaluation, and diagnostic analysis.

- **`ZAYAN_PIP_Install_Check.ipynb`**  
  Demonstration of ZAYAN using pip installation, including simple toy examples for importing the package, initializing `ZAYAN_CL`, `ZAYAN_T`, and `ZAYAN`, and running a minimal classification workflow.

### Other top-level files

- **`requirements.txt`** - Python dependencies required to run the ZAYAN package and notebooks.
- **`ZAYAN_Architecture.png`** - High-level architecture diagram of the ZAYAN framework.
- **`LICENSE`** - MIT license for this repository.
- **`README.md`** - Project overview, installation, usage instructions, and citation information.
- **`.gitignore`** - Standard Git ignore rules for Python and Jupyter projects.
- **`pyproject.toml`** - Build system and packaging metadata for installation and PyPI upload.
- **`setup.cfg`** - Optional package configuration and installation metadata, if used alongside `pyproject.toml`.

### Tested Environment

- Python 3.10.13
- torch 2.0.0+
- numpy 1.23.0+
- pandas 1.5.0+
- scikit-learn 1.2.0+
- matplotlib 3.7.0+
- optuna 3.6.0+
- jupyterlab 4.0.0+

## Installation

You can install **ZAYAN** in several ways depending on your workflow.

---

### Option 1: Clone the Repository (Recommended for Development)

```bash
git clone https://github.com/zadid6pretam/ZAYAN.git
cd ZAYAN
pip install -r requirements.txt
pip install -e .
```

### Option 2: Install Directly from GitHub (No Cloning Needed)

```bash
pip install "git+https://github.com/zadid6pretam/ZAYAN.git"
```

### Option 3: Use a Virtual Environment

```bash
python -m venv zayan-env
source zayan-env/bin/activate  # On Windows: zayan-env\Scripts\activate

git clone https://github.com/zadid6pretam/ZAYAN.git
cd ZAYAN
pip install -r requirements.txt
pip install -e .
```

### Option 4: Local Install Without Editable Mode

```bash
git clone https://github.com/zadid6pretam/ZAYAN.git
cd ZAYAN
pip install -r requirements.txt
pip install .
```

### Option 5: Install from PyPI

```bash
pip install zayan
```
