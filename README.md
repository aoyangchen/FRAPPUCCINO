# FRAPPUCCINO

Machine-learning benchmark and workflow for GT1 glycosyltransferase–acceptor reactivity prediction under novelty-controlled evaluation.

This repository accompanies the MSc thesis *Predicting Glycosyltransferase Acceptor Specificity with Variational Autoencoders and Pretrained Protein–Small-Molecule Representations*. It contains a notebook-first pipeline for dataset harmonization, feature generation, novelty-controlled benchmarking, and model evaluation across pooled-feature baseline models, token-level cross-attention, and VAE-based fusion models.

In the thesis benchmark, pooled-feature XGBoost performed best in most settings, while the early-fusion supervised VAE performed strongest under the strictest double-cold enzyme-and-substrate novelty regime.

## Why the name?

**FRAPPUCCINO** stands for:

**F**amily-1 glycosyltransferase  
**R**eactivity and  
**A**cceptor-**P**air  
**P**rediction with  
**P**retrained protein–small-molecule representations,  
**U**sing  
**C**ross-modal  
**C**ompression and  
**I**nference under  
**N**ovelty-controlled  
**O**ut-of-distribution evaluation.

## Repository structure

- `notebooks/` — end-to-end Colab workflow
- `data/` — input data and dataset documentation
- `helpers/` — reusable utility code used by the notebook
- `models/` — saved model artifacts, configs, or checkpoints (if applicable)
- `reports/` — figures, tables, and exported evaluation outputs (if applicable)

## Getting started

1. Open `notebooks/`.
2. Run the main notebook.
3. Follow the notebook cells in order to reproduce preprocessing, feature generation, benchmark construction, training, and evaluation.

The notebook is the current reference implementation of the project workflow.

## Scope

This repository focuses on binary GT1 enzyme–acceptor reactivity prediction, with evaluation across enzyme novelty, substrate novelty, and joint enzyme–substrate novelty settings.
