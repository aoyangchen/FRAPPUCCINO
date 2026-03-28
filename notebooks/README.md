# Notebooks

This folder contains the project notebooks.

## Purpose

The notebook is the main entry point for running the workflow end to end.

Use it to:

- set paths and runtime options
- run data preparation and feature generation
- run training and evaluation
- write metrics, figures, and artifacts

## Helper scripts

The notebook calls helper modules to keep repeated logic out of notebook cells:

- `nb_contracts.py` — paths, constants, and expected schemas
- `nb_drive_io.py` — Colab / Drive I/O helpers
- `nb_feature_contracts.py` — feature and split loading helpers
- `nb_model_shims.py` — training, evaluation, and artifact-writing helpers
- `nb_run_contracts.py` — run naming and output layout conventions

## Editing convention

- Edit the **notebook** for workflow order, experiment settings, and analysis output.
- Edit the **helper scripts** for reusable logic used across sections.

## Rule of thumb

Start with the notebook.  
Open a helper script only when you need the implementation behind a notebook step.
