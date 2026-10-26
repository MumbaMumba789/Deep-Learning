# IMDb Sentiment Classifier

A beginner-friendly binary sentiment classifier that uses PyTorch and a bag-of-words representation to label movie reviews as positive or negative.

## Project structure

```text
01_binary_sentiment_analysis/
├── data/
│   └── raw/IMDB Dataset.csv       # Original dataset (50,000 reviews)
├── models/
│   ├── sentiment_model.pth        # Current trained model
│   ├── vectorizer.joblib          # Matching text vocabulary
│   └── previous_notebook_run/     # Preserved original artifacts
├── notebooks/
│   └── model.ipynb                # Exploratory notebook
├── src/
│   ├── model.py                   # Shared model definition
│   ├── train.py                   # Training and validation
│   └── predict.py                 # Review prediction command
├── requirements.txt
└── README.md
```

The older notebook model and vectorizer are kept in `models/previous_notebook_run/`. The current model and vectorizer were trained together by `src/train.py`.

## Setup

This project was verified with Python 3.12. The project `.venv` is already prepared on this computer. From the project folder, use its Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## VS Code

Open this sentiment project folder itself in VS Code so its project settings apply. The settings point VS Code at `.venv` and add `src/` to Python analysis. If VS Code already remembers another interpreter, run **Python: Select Interpreter** once and choose the project `.venv`.

To recreate the environment later, install Python 3.12 and run `py -3.12 -m venv .venv` from the project folder first.

## Train the model

```powershell
.\.venv\Scripts\python.exe src\train.py
```

Training reads the CSV from `data/raw/`, reserves 20% of reviews for validation, and saves the matching model and vectorizer in `models/`. Paths are based on the project location, so the command works from any current folder.

Optional settings:

```powershell
.\.venv\Scripts\python.exe src\train.py --epochs 3 --batch-size 128 --max-features 8000
```

Defaults: 8,000 vocabulary features, 64 hidden units, two epochs, and a fixed seed. Training prints average loss and validation accuracy after each epoch.

## Predict sentiment

Pass a review as text:

```powershell
.\.venv\Scripts\python.exe src\predict.py This movie was funny and beautifully acted
```

Or omit the text to enter it interactively:

```powershell
.\.venv\Scripts\python.exe src\predict.py
```

The result includes the predicted label and positive probability. Run training first if the model files are missing or are in the older notebook format.

## Notebook

`notebooks/model.ipynb` is kept as the exploratory learning record. Its model input size now comes from the vectorizer output. The training and prediction scripts are the supported workflow for producing and loading the current saved artifacts.

## Troubleshooting

- `ModuleNotFoundError`: run `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` from the project folder.
- Missing or old model artifacts: run `.\.venv\Scripts\python.exe src\train.py`.
- Dataset missing: confirm `data/raw/IMDB Dataset.csv` exists. Training also accepts `--data PATH`.

Docker will be added after the local Python workflow is working.





