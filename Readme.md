# Detection of AI-Generated Arabic Text

## Project Overview
This project implements a full pipeline for detecting AI-generated Arabic academic abstracts.  
The system combines Arabic text preprocessing, stylometric feature extraction, and machine/deep learning models to distinguish between human-written and AI-generated texts.

## Dataset
- Source: Arabic Machine-Generated Text dataset from "The Arabic AI Fingerprint" study.
- Final size after cleaning: 36,525 rows.
- Label:
  - 1 = Human-written abstract (original_abstract)
  - 0 = AI-generated abstract (Allam, Jais, LLaMA, OpenAI)

## Methods

### Preprocessing
- Arabic text normalization (unifying letters, removing diacritics and non-Arabic symbols).
- Tokenization and stopword removal (NLTK + extended Arabic list).
- Stemming (ISRI) and lemmatization (Stanza).
- Saving the cleaned dataset as `processed_dataset.csv`.

### Feature Engineering
- Short word ratio (≤ 3 letters).
- Sentence count and line statistics.
- Foreign character ratio.
- Redundancy score based on repeated n-grams.
- TF-IDF features.
- BERT-based embeddings for contextual representation.

### Modeling
- Classical models: Logistic Regression, Naïve Bayes, SVM, Random Forest, XGBoost.
- Deep models: Feedforward Neural Network, simple DNN.
- Train/val/test split: 70% / 15% / 15%.
- Class balancing with SMOTE and feature scaling with StandardScaler.

## Results (Summary)
- TF-IDF + XGBoost achieved the best performance:
  - Accuracy: 95.10%
  - F1-score: 0.950
- TF-IDF worked best with classical models, while BERT embeddings were more suitable for neural networks.
- Stylometric analysis showed clear differences between human and AI texts in redundancy, sentence length, and vocabulary richness.

## Files
- `processed_dataset.csv` : cleaned and preprocessed data.
- Notebooks:
  - `preprocessing.ipynb`
  - `feature_engineering.ipynb`
  - `modeling_tfidf.ipynb`
  - `modeling_bert.ipynb`

### TF-IDF vs AraBERT Performance

![TF-IDF vs AraBERT Performance](./tfidf_vs_arabert.png)