# Twitter Financial News Classification

Multi-class text classification of financial news tweets into **20 categories** using classical ML baselines and deep learning models trained on ~20,000 labelled samples.

---

## Overview

This project explores the task of automatically categorising short financial news tweets. It compares traditional NLP pipelines against modern deep learning architectures, using the same preprocessed dataset throughout for a fair evaluation.

**Best result: CNN — 78.07% test accuracy / 0.7813 weighted F1**

---

## Repository Structure

```
twitter-financial-news/
├── data/
│   ├── new_data_train.csv       # Raw training data
│   ├── new_data_test.csv        # Raw test data
│   ├── train_cleaned.csv        # Cleaned/preprocessed training data
│   └── test_cleaned.csv         # Cleaned/preprocessed test data
├── models/
│   ├── cnn_model.h5             # Saved CNN model
│   ├── lstm_model.h5            # Saved LSTM model
│   ├── bilstm_model.h5          # Saved BiLSTM model
│   └── dl_models_metrics.pkl    # Saved evaluation metrics
├── preprocessed_data.pkl        # Tokenised & padded sequences + vocab
├── EDA.ipynb                    # Exploratory Data Analysis
├── training.ipynb               # Model training & evaluation
└── confusion_matrices.png       # Confusion matrix comparison
```

---

## Models & Results

### Baseline Models (TF-IDF)

| Model               | Train Acc | Val Acc | Test Acc | Weighted F1 |
|---------------------|-----------|---------|----------|-------------|
| Logistic Regression | 38.16%    | 4.58%   | 3.99%    | 0.039       |
| SVM (LinearSVC)     | 66.44%    | 6.67%   | 6.20%    | 0.069       |

> ⚠️ High train-val gap indicates both baselines overfit severely on this 20-class task with short tweet text.

---

### Deep Learning Models

| Model  | Train Acc | Val Acc | Test Acc | Test F1 |
|--------|-----------|---------|----------|---------|
| CNN    | 97.92%    | 78.67%  | **78.07%** | **0.7813** |
| BiLSTM | 95.11%    | 78.16%  | 76.97%   | 0.7715  |
| LSTM   | 91.30%    | 74.09%  | 72.84%   | 0.7317  |

**CNN is the best performing model** with the highest test accuracy and F1.

---

## Model Architectures

### CNN
```
Input (seq_len=40) → Embedding (64d) → Conv1D (32 filters, k=3) 
→ GlobalMaxPooling1D → Dense (128, ReLU) → Dropout (0.2) → Dense (20, Softmax)
Total params: 361,460
```

### LSTM
```
Input (seq_len=40) → Embedding (64d) → LSTM (128 units)
→ Dropout (0.2) → Dense (64, ReLU) → Dropout (0.2) → Dense (20, Softmax)
Total params: 456,852
```

### BiLSTM
```
Input (seq_len=40) → Embedding (64d) → Bidirectional LSTM (128 units × 2)
→ Dropout (0.2) → Dense (64, ReLU) → Dropout (0.2) → Dense (20, Softmax)
Total params: 563,860
```

---

## Training Setup

- **Dataset split:** 12,459 train / 4,153 val / 4,013 test
- **Vocabulary size:** 5,445 tokens | **Max sequence length:** 40
- **Classes:** 20 financial news categories
- **Class imbalance handling:** `class_weight='balanced'`
- **Early stopping:** patience=5, monitoring `val_loss`, restoring best weights
- **Optimiser:** Adam (lr=0.001) | **Loss:** Sparse Categorical Crossentropy
- **Batch size:** 32 | **Max epochs:** 100

---

## Getting Started

### Prerequisites

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib
```

### Run EDA

Open and run `EDA.ipynb` to explore the dataset — class distributions, tweet length stats, and token frequency analysis.

### Run Training

Open and run `training.ipynb`. It will:
1. Load `preprocessed_data.pkl` and cleaned CSVs
2. Train TF-IDF baselines (Logistic Regression + SVM)
3. Train CNN, LSTM, and BiLSTM models
4. Evaluate and compare all models
5. Save trained models to the `models/` directory

---

## Tech Stack

- **Language:** Python (Jupyter Notebook)
- **Deep Learning:** TensorFlow / Keras
- **Classical ML:** scikit-learn
- **Data:** NumPy, Pandas
- **Visualisation:** Matplotlib