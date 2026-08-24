# Detection of AI-Generated Arabic Text

## Project Overview

This project presents an end-to-end pipeline for detecting AI-generated Arabic academic abstracts.

The system combines Arabic-specific text preprocessing, stylometric feature extraction, TF-IDF representations, contextual embeddings, and machine/deep learning models to distinguish between human-written and AI-generated texts.

## Dataset

- Source: Arabic Machine-Generated Text dataset from *The Arabic AI Fingerprint* study.
- Final size after cleaning: 36,525 abstracts.
- Class distribution:
  - Human-written: 2,992 abstracts (8.2%).
  - AI-generated: 33,533 abstracts (91.8%).
- Labels:
  - `1` = Human-written abstract (`original_abstract`).
  - `0` = AI-generated abstract (ALLaM, Jais, LLaMA, and OpenAI).

## Methods

### Preprocessing

- Arabic letter normalization.
- Removal of diacritics, URLs, HTML tags, repeated punctuation, and text noise.
- Tokenization and stopword removal using NLTK and an extended Arabic stopword list.
- ISRI stemming and Stanza lemmatization.
- Saving the processed data as `processed_dataset.csv`.

### Feature Engineering

The extracted stylometric and textual features included:

- Short-word ratio (words containing three letters or fewer).
- Sentence count and structural statistics.
- Foreign-character ratio.
- Repeated n-gram redundancy score.
- Vocabulary richness using the type-token ratio.
- TF-IDF lexical features.
- BERT-based contextual embeddings.

### Modeling

Five classical machine-learning models were evaluated:

- Logistic Regression.
- Naïve Bayes.
- Support Vector Machine.
- Random Forest.
- XGBoost.

Two neural-network architectures were also evaluated:

- Feedforward Neural Network.
- Simple Deep Neural Network.

### Data Splitting and Class Balancing

- Stratified train/validation/test split: 70% / 15% / 15%.
- SMOTE was applied only to the training set to balance the minority human-written class.
- The validation and test sets retained their original class distributions.
- StandardScaler was used to normalize numerical features.

## Results

TF-IDF outperformed BERT embeddings across the classical models evaluated with both representations: Logistic Regression, SVM, and XGBoost.

The best overall configuration was TF-IDF with XGBoost:

- Accuracy: 95.11%.
- F1-score: 0.950.

The strongest BERT-based result was achieved by the Simple DNN, with an accuracy of 91.86%.

The stylometric analysis also showed measurable differences between human-written and AI-generated abstracts in sentence structure, redundancy, short-word usage, and vocabulary richness.

### TF-IDF vs BERT Performance

![TF-IDF vs BERT Performance](./tfidf_vs_arabert.png)

## Files

- `processed_dataset.csv`: Cleaned and preprocessed dataset.
- `preprocessing.ipynb`: Arabic text preprocessing pipeline.
- `feature_engineering.ipynb`: Stylometric and textual feature extraction.
- `modeling_tfidf.ipynb`: TF-IDF-based model training and evaluation.
- `modeling_bert.ipynb`: BERT-based model training and evaluation.

