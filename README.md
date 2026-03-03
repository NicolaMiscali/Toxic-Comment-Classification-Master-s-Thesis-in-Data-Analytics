#  Toxic Comment Classification — Master's Thesis in Data Analytics

A Natural Language Processing (NLP) project for **multi-label toxic comment classification**, built as part of a Master's thesis in Data Analytics. The goal is to automatically detect and categorize toxic language in online comments using classical machine learning models, with a focus on text preprocessing, feature engineering, and model comparison.

---

##  Project Overview

Online platforms face a significant challenge in moderating toxic content at scale. This project tackles that problem by building a pipeline that classifies comments into one or more of the following toxicity categories:

| Label | Description |
|---|---|
| `toxic` | General toxic language |
| `severe_toxic` | Extremely offensive content |
| `obscene` | Obscene/vulgar language |
| `threat` | Threatening statements |
| `insult` | Insulting language |
| `identity_hate` | Hate speech targeting identity groups |

The task is **multi-label**: a single comment can belong to multiple categories simultaneously.

---

##  Dataset

The dataset used is the **[Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)** from Kaggle, consisting of Wikipedia talk page comments manually labeled by human raters.

```
dati/
├── train.csv         # Labeled comments for training
├── test.csv          # Unlabeled comments for inference
└── contractions.json # Dictionary for expanding English contractions
```

---

##  Pipeline

### 1. Exploratory Data Analysis
- Label distribution analysis
- Comment length statistics
- Multi-label co-occurrence analysis

### 2. Text Preprocessing
A custom `clean_text()` function applies the following steps:
- Lowercasing
- Contraction expansion (e.g. *"won't" → "will not"*)
- Username and URL removal
- HTML entity removal
- Emoji replacement with semantic tokens (`positiveemoji` / `negativeemoji`)
- Special character removal
- Repeated character compression
- **Light stopword removal** — negations, pronouns, and modal verbs are intentionally preserved to retain semantic meaning relevant to toxicity detection

### 3. Feature Extraction
- **TF-IDF Vectorization** with:
  - `max_features = 120,000`
  - `ngram_range = (1, 2)` — unigrams and bigrams
  - `sublinear_tf = True` — log-scaled term frequencies

### 4. Stemming (variant)
A second preprocessing pipeline using **Snowball Stemmer** reduces words to their root form, enabling a direct comparison with and without stemming.

### 5. Classification Models
Both models are wrapped in a **One-vs-Rest (OvR)** strategy for multi-label classification:

- **Logistic Regression** (`max_iter=200`)
- **Multinomial Naive Bayes**

Four model configurations are evaluated in total:

| Model | Preprocessing |
|---|---|
| Logistic Regression | TF-IDF |
| Naive Bayes | TF-IDF |
| Logistic Regression + Stemming | TF-IDF + Snowball |
| Naive Bayes + Stemming | TF-IDF + Snowball |

---

##  Evaluation

Each model is evaluated using:
- **Classification Report** (Precision, Recall, F1-score per class)
- **ROC-AUC (macro)** — area under the ROC curve
- **Precision-Recall Curves** — particularly informative given the class imbalance
- **Comparative summary table** across all four configurations

### Results Summary

| Model | F1 (macro) | ROC-AUC (macro) |
|---|---|---|
| Logistic Regression | 0.46 | 0.97 |
| Naive Bayes | 0.22 | 0.89 |
| LogReg + Stemming | 0.0002 | 0.98 |
| NaiveBayes + Stemming | 0.22 | 0.89 |

> ⚠️ Fill in the actual metric values from your notebook output before publishing.

---

##  Additional Analysis

Beyond classification, the project includes qualitative exploration of toxic language patterns:

- **Word Clouds** — visual representation of the most frequent words per toxicity class
- **Top-20 most common words** per class
- **Top bigrams** per class
- **LDA Topic Modeling** — latent topic discovery within each toxicity category (4 topics per class)

---

##  Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| pandas / numpy | Data manipulation |
| scikit-learn | ML models, TF-IDF, evaluation |
| NLTK | Stopwords, Snowball Stemmer |
| matplotlib / seaborn | Visualization |
| WordCloud | Text visualization |
| re (regex) | Text cleaning |

---

##  Author

**[Nicola Miscali]**  
Master's in Data Analytics  
[Università degli Studi Roma Tre] — Academic Year [2024/2025]

---

## 📄 License

This project is for academic purposes. The dataset is subject to the [Jigsaw/Kaggle competition terms](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/rules).
