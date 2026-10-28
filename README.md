# PyTorch Learning Lab

A personal workspace for learning Python and PyTorch through small lessons and end-to-end projects. The folder is named **tensor flow**, but the notebooks and projects here use PyTorch.

## Start here

Open this folder in VS Code, then begin with the numbered notebooks in `myfirstlesson_on_nn/notebooks/`. Work through them in order; they move from tensor basics to real data, neural networks, and classification.

## What is where

| Folder | Contents |
| --- | --- |
| `myfirstlesson_on_nn/notebooks/` | Numbered PyTorch lessons and their helper functions |
| `myfirstlesson_on_nn/data/` | White-wine quality and concrete-strength datasets used by the lessons |
| `projects/01_binary_sentiment_analysis/` | IMDb sentiment project with training and prediction scripts; see its [project guide](projects/01_binary_sentiment_analysis/README.md) |
| `projects/02_heart_disease_prediction/` | Heart-disease classification notebook |

## Working with the notebooks

1. In VS Code, open a notebook and select a Python kernel that has PyTorch, pandas, scikit-learn, and Jupyter installed. The concrete-strength lesson also needs `xlrd` to read its Excel file.
2. Run the cells from top to bottom. The notebooks use paths relative to their own folder, so keep each notebook beside its helper files and leave the datasets in `myfirstlesson_on_nn/data/`.
3. The heart-disease notebook reads its dataset from the source URL in the notebook and needs an internet connection for that cell.

This repository has a root `.venv` and the sentiment project has its own `.venv`. If VS Code selects the wrong interpreter, use **Python: Select Interpreter** and choose the environment you want for that notebook.

## Keep the workspace tidy

Datasets, trained model files, and the MLflow tracking database are kept with the lesson or project that uses them. Python caches, notebook checkpoints, virtual environments, and MLflow run output are local working files and are excluded from the VS Code file view or Git as appropriate.
