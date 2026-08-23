# Stock Sentiment Movement Predictor

Predicts stock price movement (up/down) from financial news headlines using
sentiment and text-classification models. Four modeling approaches —
**FinBERT**, **DistilBERT**, **LightGBM**, and an **ANN (TF-IDF based)** —
are each implemented and benchmarked independently to compare how
transformer-based embeddings stack up against classical/traditional
ML pipelines for headline-driven stock movement prediction.

---

## How It Works

Each notebook follows the same general pipeline, with a different model
at the core:

1. **Load data** — Daily stock news headlines paired with a movement
   label (e.g. price up/down for that day).
2. **Preprocess text** — Combine/clean headline columns; for the classical
   models (LightGBM, ANN), text is also lowercased, stopwords removed, and
   lemmatized (NLTK).
3. **Feature extraction**
   - **FinBERT / DistilBERT** — headlines are passed through the
     pretrained transformer to generate contextual embeddings
     (`[CLS]` token representation).
   - **LightGBM / ANN** — headlines are vectorized using **TF-IDF**
     (optionally reduced with TruncatedSVD).
4. **Train/test split** — 80/20 split via `train_test_split`.
5. **Classification**
   - FinBERT embeddings → Logistic Regression classifier
   - DistilBERT → fine-tuned `DistilBertForSequenceClassification`
   - LightGBM → `LGBMClassifier` on TF-IDF features
   - ANN → Keras `Sequential` dense network on TF-IDF features
6. **Evaluation** — Accuracy, classification report, and confusion matrix
   for each model, so results are directly comparable across the four
   notebooks.

---

## Models Compared

| Model | Approach | File |
|---|---|---|
| **FinBERT** | Domain-specific financial transformer embeddings + Logistic Regression | `1787489638952_FINBERT__1_.ipynb` |
| **DistilBERT** | Fine-tuned general-purpose transformer classifier | `1787489609847_DistilBERT.ipynb` |
| **LightGBM** | Gradient-boosted trees on TF-IDF features | `1787489609848_LIGHTGBM__1_.ipynb` |
| **ANN** | Feedforward neural network on TF-IDF features | `1787489638952_ANN.ipynb` |

---

## Requirements

- Python 3.8+
- Google Colab (recommended) or a local Jupyter environment with GPU
  access (recommended for FinBERT/DistilBERT — CPU works but is slower)

### Dependencies
```
pandas
numpy
scikit-learn
matplotlib
seaborn
nltk
joblib
lightgbm
tensorflow
torch
transformers
tqdm
```

Each notebook installs its own dependencies in the first cell, so no
separate install step is required if you're running in Colab.

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/yakkalakarthikeya/Stock-Sentiment-Movement-Predictor.git
cd Stock-Sentiment-Movement-Predictor
```

### 2. (Local only) Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk joblib lightgbm tensorflow torch transformers tqdm
```

### 3. Add the dataset
Each notebook expects a CSV of stock headlines with a movement label,
e.g.:
```
/content/Stock News Dataset.csv
/content/Sentiment_Stock_data.csv
```
Place your dataset in the same directory as the notebook (or update the
`file_path` / `CSV_PATH` variable at the top of each notebook) before
running.

---

## Run

Each model lives in its own self-contained notebook — open and run
whichever one you want to test:

```bash
jupyter notebook 1787489638952_FINBERT__1_.ipynb
jupyter notebook 1787489609847_DistilBERT.ipynb
jupyter notebook 1787489609848_LIGHTGBM__1_.ipynb
jupyter notebook 1787489638952_ANN.ipynb
```

Or upload any notebook directly to **Google Colab** and run all cells
top to bottom — each one installs its own dependencies automatically.

**Recommended run order:** LightGBM → ANN → FinBERT → DistilBERT
(fastest to slowest, since FinBERT/DistilBERT require generating
transformer embeddings or fine-tuning).

Each notebook prints:
- Sample predictions on the first few rows
- Overall accuracy
- Classification report (precision/recall/F1)
- Confusion matrix

---

## Project Structure

```
Stock-Sentiment-Movement-Predictor/
├── 1787489638952_FINBERT__1_.ipynb     # FinBERT embeddings + Logistic Regression
├── 1787489609847_DistilBERT.ipynb      # Fine-tuned DistilBERT classifier
├── 1787489609848_LIGHTGBM__1_.ipynb    # TF-IDF + LightGBM
├── 1787489638952_ANN.ipynb             # TF-IDF + Keras ANN
└── README.md
```
