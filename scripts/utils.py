# utils.py
# Helper functions and common utilities used across different modules.

import os
import re
import gc
import nltk
import joblib
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from scipy import sparse as sp

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================================
# NLTK RESOURCES
# ==============================================================================

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================

def setup_environment():
    """
    Configures the plotting environment by downloading necessary NLTK data,
    setting up styles, and applying a suitable Arabic font.
    """
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
   
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.unicode_minus"] = False
    arabic_fonts = ["Arial", "Amiri", "DejaVu Sans", "Tahoma"]
    font_found = False
    for font in arabic_fonts:
        try:
            font_path = findfont(FontProperties(family=font))
            if os.path.exists(font_path):
                plt.rcParams["font.family"] = font
                print(f"Using Matplotlib font: {font}")
                font_found = True
                break
        except:
            continue
    if not font_found:
        print("Warning: No suitable Arabic font found for Matplotlib.")
        plt.rcParams["font.family"] = "sans-serif"


def get_arabic_font_path() -> str | None:
    """
    Searches for a suitable Arabic font file path for WordCloud.
    """
    arabic_fonts = ["Arial", "Amiri", "DejaVu Sans", "Tahoma"]
    for font in arabic_fonts:
        try:
            font_path = findfont(FontProperties(family=font))
            if os.path.exists(font_path):
                return font_path
        except:
            continue
    print("Warning: No font file found for WordCloud.")
    return None


# ==============================================================================
# FEATURE SCALING
# ==============================================================================

def improve_features(X_train, X_val, X_test):
    """
    Scale features using StandardScaler.
    Handles both dense and sparse input matrices.
    """
    if sp.issparse(X_train):
        scaler = StandardScaler(with_mean=False)
    else:
        scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled


# ==============================================================================
# MODEL PERSISTENCE
# ==============================================================================
def save_model(model, model_name):
    """
    Save a trained model to disk using joblib.
    Creates 'trained_models' directory if it doesn't exist.
    Prints the full absolute path where the model is saved.
    """
    # Create directory
    os.makedirs("trained_models", exist_ok=True)
    
    # Define file path
    file_path = os.path.join("trained_models", f"{model_name}.joblib")
    
    # Save the model
    joblib.dump(model, file_path)
    
    # Get absolute path
    abs_path = Path(file_path).resolve()
    
    # Print confirmation with full path
    print(f"Model '{model_name}' saved successfully!")
    print(f"Location: {abs_path}")
    
    return str(abs_path)  # Return absolute path as string

# ==============================================================================
# STRATIFIED SAMPLING
# ==============================================================================

def stratified_sample(X, y, n_samples=4000):
    """
    Perform stratified sampling to reduce dataset size while preserving class distribution.
    """
    if len(y) <= n_samples:
        return X, y
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=n_samples/len(y), random_state=42)
    _, idx = next(splitter.split(X, y))
    return X[idx], y[idx]


# ==============================================================================
# EMBEDDINGS GENERATION (SENTENCE TRANSFORMERS)
# ==============================================================================

def generate_embeddings(model, texts, l2_normalize=True):
    """
    Generate embeddings using a SentenceTransformer model.
    Optionally applies L2 normalization.
    """
    embs = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    if l2_normalize:
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs = embs / norms
    return embs
