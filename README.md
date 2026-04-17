# Enterprise Text Analytics & Classification Suite

## Overview
This repository contains a comprehensive suite of Natural Language Processing (NLP) tools designed for text preprocessing, feature extraction, statistical analysis, and machine learning classification. A primary application of this suite is the detection of spam and phishing content, supplemented by model explainability (XAI) using SHAP.

## Core Capabilities

### 1. Text Preprocessing & Cleaning
Robust text normalization pipelines to prepare raw text for downstream machine learning tasks.
- **Modules:** Punctuation removal, regex-based tokenization, and stopword filtering.
- **Key Files:** `practice/Basic_stats.py`

### 2. Feature Extraction & Embeddings
Various techniques to convert textual data into machine-readable numerical formats.
- **TF-IDF (Term Frequency-Inverse Document Frequency):** Scripts for vectorizing corpus data and determining feature significance.
  - **Key Files:** `tf-idf.py`, `practice/tf_idf_practice.py`
- **Word Embeddings (Word2Vec & GloVe):** Implementation of custom skip-gram models, utilizing pre-trained GloVe embeddings, and training custom Word2Vec models using `gensim`.
  - **Key Files:** `word2vector.py`, `Word2Vectors.py`
- **Statistical NLP:** Pointwise Mutual Information (PMI), N-grams, and corpus reading.
  - **Key Files:** `PMI.py`, `reading_files.py`

### 3. Machine Learning Classification
Supervised learning models tuned for text classification tasks such as spam and phishing detection.
- **Models:** Random Forest Classifier, Logistic Regression.
- **Metrics:** Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.
- **Key Files:** `adv_tfidf_rf.py`

### 4. Explainable AI (XAI)
Transparency and bias detection for ML classification models to ensure interpretable predictions.
- **SHAP Integration:** Summary plots illustrating feature impact on phishing detection models.
- **Key Files:** `practice/Bias_shap.py`

## Installation & Requirements

Ensure you have Python 3.8+ installed. Install the required dependencies using `pip`:

```bash
pip install pandas numpy scikit-learn nltk spacy gensim shap matplotlib
```

*Note: You will also need to download necessary NLTK corpora and Spacy models prior to running certain scripts:*
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## Data Requirements
Several scripts (e.g., `adv_tfidf_rf.py`, `word2vector.py`) expect a dataset named `spam.csv` located in the root or `practice/` directory. Ensure your dataset is properly formatted (typically standard SMS Spam Collection formatting with `v1` as the label and `v2` as the text).

## Usage Example

To run the Random Forest TF-IDF classification pipeline:

```bash
python adv_tfidf_rf.py
```

To generate SHAP explainability plots for the Logistic Regression phishing detector:

```bash
python practice/Bias_shap.py
```

## License
Proprietary / Internal Use Only.