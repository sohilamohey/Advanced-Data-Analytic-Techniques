# data_preparation.py
# Functions for data cleaning, preprocessing, and feature engineering.

import os
import re
import gc
import nltk
import logging
import warnings
import numpy as np
import pandas as pd
import stanza
import pickle
import torch
from tqdm import tqdm
from tabulate import tabulate
from scipy import sparse
from pyarabic import araby
from bidi.algorithm import get_display
import arabic_reshaper
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.isri import ISRIStemmer
from nltk.util import ngrams
from collections import Counter
from wordcloud import WordCloud
from matplotlib.font_manager import FontProperties, findfont
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


# ==============================================================================
# NLTK RESOURCES
# ==============================================================================

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)


# ==============================================================================
# GLOBAL CONFIGURATION & HELPERS
# ==============================================================================

# Arabic stopwords (NLTK + extended list)
arabic_stopwords = set(stopwords.words('arabic'))
try:
    import requests
    url = 'https://raw.githubusercontent.com/mohataher/arabic-stop-words/master/list.txt'
    response = requests.get(url)
    if response.status_code == 200:
        arabic_stopwords.update(set(response.text.splitlines()))
except Exception:
    pass

# PyArabic diacritics fallback
try:
    from pyarabic import araby
    _HAS_PYARABIC = True
except Exception:
    _HAS_PYARABIC = False
    _DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670]")

def _strip_tashkeel_safe(text: str) -> str:
    """Strip diacritics using pyarabic or regex fallback."""
    if _HAS_PYARABIC:
        return araby.strip_tashkeel(text)
    return _DIACRITICS_RE.sub("", text)

# Arabic pattern
arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')

# Logging
logging.basicConfig(filename='preprocessing_errors.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ==============================================================================
# DATA LOADING & PREPARATION
# ==============================================================================

def setup_directory(data_dir):
    """Create directory if it doesn't exist."""
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def load_datasets(data_dir):
    """Load datasets from local files or Hugging Face."""
    file_paths = {
        "by_polishing": os.path.join(data_dir, "by_polishing.parquet"),
        "from_title": os.path.join(data_dir, "from_title.parquet"),
        "from_title_and_content": os.path.join(data_dir, "from_title_and_content.parquet")
    }
   
    if all(os.path.exists(p) for p in file_paths.values()):
        print("Loading dataset from local files...")
        return (
            pd.read_parquet(file_paths["by_polishing"]),
            pd.read_parquet(file_paths["from_title"]),
            pd.read_parquet(file_paths["from_title_and_content"])
        )
   
    print("Downloading dataset from Hugging Face...")
    dataset = load_dataset("KFUPM-JRCAI/arabic-generated-abstracts")
    by_polishing = pd.DataFrame(dataset["by_polishing"])
    from_title = pd.DataFrame(dataset["from_title"])
    from_title_and_content = pd.DataFrame(dataset["from_title_and_content"])
   
    by_polishing.to_parquet(os.path.join(data_dir, "by_polishing.parquet"))
    from_title.to_parquet(os.path.join(data_dir, "from_title.parquet"))
    from_title_and_content.to_parquet(os.path.join(data_dir, "from_title_and_content.parquet"))
    print(f"DataFrames saved as Parquet files in: {data_dir}")
   
    return by_polishing, from_title, from_title_and_content


def reshape_dataframe(df, var_name='category', value_name='text'):
    """Reshape DataFrame using pd.melt."""
    return pd.melt(df, var_name=var_name, value_name=value_name)


def save_to_excel(dfs, sheet_names, output_file):
    """Save DataFrames to Excel with different sheets, ensuring Arabic support."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        for df, sheet_name in zip(dfs, sheet_names):
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            worksheet.right_to_left()
            format_rtl = writer.book.add_format({'align': 'right'})
            for col_num, col_name in enumerate(df.columns):
                worksheet.set_column(col_num, col_num, None, format_rtl)
    print(f"Data saved to: {output_file} (with Arabic RTL formatting)")


def concatenate_and_save(dfs, output_file):
    """Concatenate DataFrames, save to Excel with Arabic support, and return the result."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    concatenated_data = pd.concat(dfs, ignore_index=True)
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        concatenated_data.to_excel(writer, sheet_name='Concatenated_Data', index=False)
        worksheet = writer.sheets['Concatenated_Data']
        worksheet.right_to_left()
        format_rtl = writer.book.add_format({'align': 'right'})
        for col_num, col_name in enumerate(concatenated_data.columns):
            worksheet.set_column(col_num, col_num, None, format_rtl)
    print(f"Concatenated data saved to: {output_file} (with Arabic RTL formatting)")
    return concatenated_data


def DataPreparation():
    """Main function to process datasets and return concatenated DataFrame."""
    os.chdir("F:\\Advanced Data Analytic Techniques")
    data_dir = setup_directory("./arabic_dataset")
   
    by_polishing, from_title, from_title_and_content = load_datasets(data_dir)
   
    by_polishing_long = reshape_dataframe(by_polishing)
    from_title_and_content_long = reshape_dataframe(from_title_and_content)
    from_title_long = reshape_dataframe(from_title)
   
    dfs = [by_polishing_long, from_title_and_content_long, from_title_long]
    sheet_names = ['by_polishing', 'from_title_and_content', 'from_title']
    save_to_excel(dfs, sheet_names, os.path.join(data_dir, 'reshaped_datasets.xlsx'))
   
    concatenated_df = concatenate_and_save(dfs, os.path.join(data_dir, 'concatenated_data.xlsx'))
    return concatenated_df


# ==============================================================================
# DATA EXPLORATION & QUALITY
# ==============================================================================

def inspect_dataset_structure(df: pd.DataFrame):
    """Inspects and prints the basic structure of the dataset."""
    print("\n================================================= Task 1.3 (Part 1): Inspect Dataset Structure =================================================")
    print("Columns and Data Types:")
    print(df.dtypes)
    print(f"\n=================================================\nShape: {df.shape}")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows and missing values, then reset index."""
    print("\n=== Data Cleaning ===")
    print("Duplicate Rows Before Cleaning:", int(df.duplicated().sum()))
    cleaned_df = df.drop_duplicates().dropna().reset_index(drop=True)
    print("Duplicate Rows After Cleaning:", int(cleaned_df.duplicated().sum()))
    print("Rows After Removing Missing Values:", cleaned_df.shape)
    return cleaned_df


def add_category_encode(df: pd.DataFrame,
                        source_col: str = 'category',
                        target_col: str = 'category_encode') -> pd.DataFrame:
    print("\n=== Building 'category_encode' from 'category' ===")
    if source_col not in df.columns:
        raise ValueError(f"Column '{source_col}' not found in DataFrame.")
    s = df[source_col].astype(str).str.strip().str.lower()
    df[target_col] = (s == 'original_abstract') | (s == 'orginal_abstract')
    df[target_col] = df[target_col].astype(int)
    print("'category_encode' counts (1=original_abstract, 0=generated/other):")
    print(df[target_col].value_counts(dropna=False).sort_index())
    return df


def _plot_two_pies_side_by_side(left_vals, left_labels, left_title,
                                right_vals, right_labels, right_title,
                                main_title):
    """Two compact, readable pies side-by-side (no crowded text)."""
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'domain'}, {'type': 'domain'}]],
        subplot_titles=(left_title, right_title),
        horizontal_spacing=0.14
    )
    fig.add_trace(
        go.Pie(
            labels=left_labels,
            values=left_vals,
            textinfo='percent',
            textposition='inside',
            insidetextorientation='radial',
            hovertemplate='%{label}: %{value} (%{percent})<extra></extra>',
            marker=dict(line=dict(color='white', width=1)),
            showlegend=True,
            sort=False
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Pie(
            labels=right_labels,
            values=right_vals,
            textinfo='percent',
            textposition='inside',
            insidetextorientation='radial',
            hovertemplate='%{label}: %{value} (%{percent})<extra></extra>',
            marker=dict(line=dict(color='white', width=1)),
            showlegend=True,
            sort=False
        ),
        row=1, col=2
    )
    fig.update_layout(
        title_text=main_title,
        title_font_size=16,
        width=1000, height=340,
        margin=dict(l=20, r=140, t=60, b=20),
        legend=dict(orientation='v',
                    yanchor='middle', y=0.5,
                    xanchor='left', x=1.02,
                    font=dict(size=11)),
        uniformtext_minsize=9,
        uniformtext_mode='hide'
    )
    for ann in fig['layout']['annotations']:
        ann['font'] = dict(size=12)
    fig.show()


def _target_counts_and_title(df: pd.DataFrame):
    if 'label' in df.columns:
        s = df['label'].astype(str)
        return s.value_counts(dropna=False), "Target Distribution — Label"
    elif 'category_encode' in df.columns:
        s = df['category_encode'].map({1: 'Original (1)', 0: 'Generated (0)'})
        return s.value_counts(dropna=False), "Target Distribution — Binary (from category)"
    elif 'category' in df.columns:
        s = df['category'].astype(str)
        return s.value_counts(dropna=False), "Target Distribution — Category Breakdown"
    else:
        raise ValueError("No 'label' or 'category' or 'category_encode' column found.")


def _orig_vs_gen_counts(df: pd.DataFrame):
    if 'category_encode' in df.columns:
        s = df['category_encode'].astype(str)
        original = int((s == 1).sum())
        generated = int((s == 0).sum())
        return {'Original': original, 'Generated': generated}
    if 'category' in df.columns:
        s = df['category'].astype(str).str.strip().str.lower()
        original = int((s == 'original_abstract').sum() + (s == 'orginal_abstract').sum())
        generated = int(len(s) - original)
        return {'Original': original, 'Generated': generated}
    return {'Original': 0, 'Generated': 0}


def plot_distributions_before_after(df_before: pd.DataFrame, df_after: pd.DataFrame):
    """Show before/after cleaning distributions: (left) Target label, (right) Original vs Generated."""
    target_counts_b, target_title_b = _target_counts_and_title(df_before)
    og_b = _orig_vs_gen_counts(df_before)
    print("\n=== BEFORE cleaning ===")
    print("Target counts:")
    print(target_counts_b)
    print("Original vs Generated:", og_b)
    _plot_two_pies_side_by_side(
        left_vals=target_counts_b.values,
        left_labels=target_counts_b.index.astype(str),
        left_title=target_title_b,
        right_vals=list(og_b.values()),
        right_labels=list(og_b.keys()),
        right_title="Original vs Generated",
        main_title="Target Variable Distributions — BEFORE Cleaning"
    )
    target_counts_a, target_title_a = _target_counts_and_title(df_after)
    og_a = _orig_vs_gen_counts(df_after)
    print("\n=== AFTER cleaning ===")
    print("Target counts:")
    print(target_counts_a)
    print("Original vs Generated:", og_a)
    _plot_two_pies_side_by_side(
        left_vals=target_counts_a.values,
        left_labels=target_counts_a.index.astype(str),
        left_title=target_title_a,
        right_vals=list(og_a.values()),
        right_labels=list(og_a.keys()),
        right_title="Original vs Generated",
        main_title="Target Variable Distributions — AFTER Cleaning"
    )


def check_text_column_quality(df: pd.DataFrame, text_columns: list) -> dict:
    """Checks for inconsistencies in specified text columns."""
    text_quality_results = {
        'empty_strings': {},
        'high_special_char_strings': {},
        'short_strings': {}
    }
    special_char_pattern = re.compile(r'[^\u0600-\u06FFa-zA-Z0-9\s]')
    for col in text_columns:
        if col in df.columns:
            s = df[col].astype("string")
            empty_count = s.fillna("").str.strip().eq("").sum()
            text_quality_results['empty_strings'][col] = int(empty_count)
            strings = s.fillna("")
            lengths = strings.str.len().where(lambda x: x > 0, other=1)
            specials = strings.apply(lambda x: len(special_char_pattern.findall(x)))
            high_special_char_count = (specials / lengths > 0.5).sum()
            text_quality_results['high_special_char_strings'][col] = int(high_special_char_count)
            short_string_count = strings.str.strip().str.len().between(1, 3).sum()
            text_quality_results['short_strings'][col] = int(short_string_count)
        else:
            print(f"Warning: Text column '{col}' not found in DataFrame.")
    print("\n=====================================================\nText Quality Check Results:")
    print("=" * 40)
    has_issues = False
    for col in text_columns:
        if col in df.columns:
            print(f"\nColumn: '{col}'")
            print(f" Empty strings: {text_quality_results['empty_strings'][col]}")
            print(f" High special character strings: {text_quality_results['high_special_char_strings'][col]}")
            print(f" Short strings (<=3 chars): {text_quality_results['short_strings'][col]}")
            if (text_quality_results['empty_strings'][col] > 0 or
                text_quality_results['high_special_char_strings'][col] > 0 or
                text_quality_results['short_strings'][col] > 0):
                has_issues = True
    if not has_issues:
        print("\nNo text quality issues found in any column!")
    else:
        print("\nText quality issues found in some columns.")
    return text_quality_results


def assess_data_quality(df: pd.DataFrame) -> dict:
    """Performs a comprehensive assessment of data quality."""
    print("\n================================================= Task 1.3 (Part 3): Assess Data Quality =================================================")
    print("\nMissing Values:")
    missing_values = df.isnull().sum()
    print(missing_values)
    print("\nConverting Columns to String")
    cols_to_convert = [c for c in ['category', 'text'] if c in df.columns]
    if cols_to_convert:
        df = df.astype({c: 'string' for c in cols_to_convert})
    print(df.info())
    print("\n=================================================\nDuplicate Rows:")
    duplicates = int(df.duplicated().sum())
    print(duplicates)
    text_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()
    if text_columns:
        print("\nInconsistencies in Text Columns:")
        print(f"Text columns found: {text_columns}")
        text_quality_results = check_text_column_quality(df, text_columns)
    else:
        print("\nNo text columns found for inconsistency check.")
        text_quality_results = {
            'empty_strings': {},
            'high_special_char_strings': {},
            'short_strings': {}
        }
    return {
        'missing_values': missing_values.to_dict(),
        'duplicates': duplicates,
        'text_column_inconsistencies': text_quality_results
    }


# ==============================================================================
# TEXT PREPROCESSING PIPELINE
# ==============================================================================

def setup_environment():
    """Configures the plotting environment."""
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
    for font in arabic_fonts:
        try:
            font_path = findfont(FontProperties(family=font))
            if os.path.exists(font_path):
                plt.rcParams["font.family"] = font
                print(f"Using Matplotlib font: {font}")
                break
        except:
            continue
    else:
        print("Warning: No suitable Arabic font found for Matplotlib.")
        plt.rcParams["font.family"] = "sans-serif"


def get_arabic_font_path() -> str | None:
    """Searches for a suitable Arabic font file path for WordCloud."""
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


def arabic_text_preprocessing_pipeline(text: str, stop_words: set) -> str:
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = re.sub(r"[أإآ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة\b", "ه", text)
    text = _strip_tashkeel_safe(text)
    text = re.sub(r"[^\u0600-\u06FF\s]", "", text)
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
    stemmer = ISRIStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    return " ".join(stemmed_words)


def advanced_arabic_text_cleaning(text: str, synonym_dict: dict = None, dialect_dict: dict = None) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'([؟!.,])\1+', r'\1', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if dialect_dict:
        words = text.split()
        text = " ".join([dialect_dict.get(word, word) for word in words])
    if synonym_dict:
        words = text.split()
        text = " ".join([synonym_dict.get(word, word) for word in words])
    return text


def arabic_text_lemmatization_pipeline(text: str, stop_words: set) -> str:
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = re.sub(r"[أإآ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة\b", "ه", text)
    text = _strip_tashkeel_safe(text)
    text = re.sub(r"[^\u0600-\u06FF\s]", "", text)
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
    try:
        doc = nlp(" ".join(filtered_words))
        lemmatized_words = [word.lemma for sent in doc.sentences for word in sent.words if word.lemma]
    except Exception as e:
        logging.error(f"Error in lemmatization for text: '{text[:50]}...' - {str(e)}")
        lemmatized_words = filtered_words
    return " ".join(lemmatized_words)


# ==============================================================================
# STATISTICAL ANALYSIS
# ==============================================================================

def calculate_average_word_length(text: str) -> float:
    """Calculates the average word length in a given text."""
    words = word_tokenize(text) if isinstance(text, str) else []
    if not words:
        return 0.0
    return sum(len(word) for word in words) / len(words)


def calculate_average_sentence_length(text: str) -> float:
    """Calculates the average sentence length in a given text."""
    sentences = sent_tokenize(text) if isinstance(text, str) else []
    if not sentences:
        return 0.0
    sentence_lengths = [len(word_tokenize(s)) for s in sentences]
    return sum(sentence_lengths) / len(sentence_lengths)


def calculate_type_token_ratio(text: str) -> float:
    """Calculates the Type-Token Ratio (TTR) for a given text."""
    tokens = word_tokenize(text) if isinstance(text, str) else []
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def perform_statistical_analysis(df: pd.DataFrame):
    """Calculates and prints statistical features of the text."""
    print("\n================================================= Task 2.2 (Part 1): Statistical Analysis =================================================")
    df['word_count'] = df['processed_text'].apply(lambda x: len(word_tokenize(x)) if isinstance(x, str) else 0)
    df['avg_word_length'] = df['processed_text'].apply(calculate_average_word_length)
    df['sentence_count'] = df['text'].apply(lambda x: len(sent_tokenize(x)) if isinstance(x, str) else 0)
    df['avg_sentence_length'] = df.apply(
        lambda row: (row['word_count'] / row['sentence_count']) if row['sentence_count'] > 0 else 0.0,
        axis=1
    )
    df['ttr'] = df['processed_text'].apply(calculate_type_token_ratio)
    summary_df = df.groupby('category_name', dropna=False).agg(
        Count=('category_name', 'size'),
        Avg_Word_Length=('avg_word_length', 'mean'),
        Avg_Sentence_Length=('avg_sentence_length', 'mean'),
        Vocabulary_Richness_TTR=('ttr', 'mean')
    ).reset_index()
    print("\n==================================================")
    print("Category Comparison Summary Table")
    print("==================================================")
    print("This table summarizes statistical metrics (count, average word length, average sentence length, and vocabulary richness via Type-Token Ratio) for each category.")
    print(tabulate(summary_df, headers='keys', tablefmt='psql', showindex=False))


# ==============================================================================
# LEXICAL ANALYSIS
# ==============================================================================

def reshape_arabic_text(text):
    """Helper function to reshape Arabic text for proper display."""
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)


def get_proper_arabic_function_words():
    """Returns a comprehensive set of true Arabic function words"""
    return {
        'من', 'إلى', 'عن', 'على', 'في', 'ب', 'ك', 'ل', 'ت', 'وا',
        'و', 'ثم', 'او', 'بل', 'لكن', 'الا', 'لا',
        'هل', 'ما', 'متى', 'أين', 'كيف', 'لماذا',
        'إذا', 'لو', 'لولا', 'لوما', 'اما',
        'ها', 'ك', 'ني', 'نا', 'كم', 'كن', 'هم', 'هن',
        'لا', 'ما', 'لم', 'لن', 'ليس',
        'أن', 'إن', 'كي', 'لان', 'اذ', 'اذا',
        'قد', 'سوف', 'كل', 'بعض', 'أي', 'هذا', 'هذه', 'ذلك', 'تلك'
    }


def analyze_function_words(texts):
    """Analyze true function words in a list of texts."""
    function_words = get_proper_arabic_function_words()
    func_word_counts = {}
    for text in texts:
        if isinstance(text, str):
            words = text.split()
            for word in words:
                clean_word = word.strip('.,،:;!?()[]{}"\'')
                if clean_word in function_words:
                    func_word_counts[clean_word] = func_word_counts.get(clean_word, 0) + 1
    return func_word_counts


def analyze_punctuation(texts):
    """Analyze punctuation in a list of texts."""
    punctuation_marks = {'.', '،', ':', '!', '(', ')', '[', ']', '{', '}', ';', '؟', '"', "'"}
    punct_counts = {}
    for text in texts:
        if isinstance(text, str):
            for char in text:
                if char in punctuation_marks:
                    punct_counts[char] = punct_counts.get(char, 0) + 1
    return punct_counts


def analyze_specific_terms(texts, terms):
    """Analyze specific terms in a list of texts."""
    term_counts = {term: 0 for term in terms}
    for text in texts:
        if isinstance(text, str):
            for term in terms:
                term_counts[term] += text.count(term)
    return term_counts


def perform_lexical_analysis(df: pd.DataFrame):
    """Performs and prints lexical analysis results with proper Arabic text formatting."""
    print("\n--- Lexical Analysis: Comparing the use of function words, punctuation, and specific terms ---")
    lexical_results = []
    categories = sorted(df['category_name'].unique())
    print("\n" + "="*80)
    print("1. FUNCTION WORD ANALYSIS (Using Original Text)")
    print("="*80)
    all_func_words = set()
    func_word_counts_by_category = {}
    for category in categories:
        category_texts = df[df['category_name'] == category]['text']
        func_word_counts = analyze_function_words(category_texts)
        func_word_counts_by_category[category] = func_word_counts
        all_func_words.update(func_word_counts.keys())
        top_words = sorted(func_word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\nTop 10 function words for {reshape_arabic_text(category)}:")
        if top_words:
            for i, (word, count) in enumerate(top_words, 1):
                print(f" {i:2d}. {reshape_arabic_text(word):<10} : {count:>6}")
        else:
            print(" No function words found.")
    if all_func_words:
        print(f"\nComparative Function Word Frequencies (Top 10 Most Common):")
        print("-" * 60)
        all_func_words_list = sorted(all_func_words,
                                   key=lambda x: sum(func_word_counts_by_category[cat].get(x, 0) for cat in categories),
                                   reverse=True)[:10]
        for word in all_func_words_list:
            print(f"\n{reshape_arabic_text(word)}:")
            for cat in categories:
                count = func_word_counts_by_category.get(cat, {}).get(word, 0)
                print(f" {reshape_arabic_text(cat):<20}: {count:>6}")
                lexical_results.append({
                    'Category': cat,
                    'Type': 'Function Word',
                    'Term': word,
                    'Count': count
                })
    else:
        print("\nNo function words found in any category.")
    print("\n" + "="*80)
    print("2. PUNCTUATION ANALYSIS")
    print("="*80)
    all_punctuations = set()
    punct_counts_by_category = {}
    for category in categories:
        category_texts = df[df['category_name'] == category]['text']
        punct_counts = analyze_punctuation(category_texts)
        punct_counts_by_category[category] = punct_counts
        all_punctuations.update(punct_counts.keys())
    for category in categories:
        punct_counts = punct_counts_by_category.get(category, {})
        top_punct = sorted(punct_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\nTop 5 punctuation marks for {reshape_arabic_text(category)}:")
        if top_punct:
            for i, (punct, count) in enumerate(top_punct, 1):
                print(f" {i}. '{punct}' : {count}")
        else:
            print(" No punctuation marks found.")
    if all_punctuations:
        print(f"\nComparative Punctuation Usage (Top 5 Most Common):")
        print("-" * 50)
        top_punctuations = sorted(all_punctuations,
                                key=lambda x: sum(punct_counts_by_category[cat].get(x, 0) for cat in categories),
                                reverse=True)[:5]
        for punct in top_punctuations:
            print(f"\n'{punct}':")
            for cat in categories:
                count = punct_counts_by_category.get(cat, {}).get(punct, 0)
                print(f" {reshape_arabic_text(cat):<20}: {count:>6}")
                lexical_results.append({
                    'Category': cat,
                    'Type': 'Punctuation',
                    'Term': punct,
                    'Count': count
                })
    else:
        print("\nNo punctuation marks found in any category.")
    print("\n" + "="*80)
    print("3. SPECIFIC TERM ANALYSIS")
    print("="*80)
    specific_terms = ['اقتصاد', 'أندلس', 'غزو']
    print(f"Analyzing specific terms: {', '.join([reshape_arabic_text(t) for t in specific_terms])}")
    for term in specific_terms:
        print(f"\n{reshape_arabic_text(term)}:")
        print("-" * 40)
        for category in categories:
            category_texts = df[df['category_name'] == category]['text']
            term_counts = analyze_specific_terms(category_texts, [term])
            count = term_counts.get(term, 0)
            print(f" {reshape_arabic_text(category):<20}: {count:>6} occurrences")
            lexical_results.append({
                'Category': category,
                'Type': 'Specific Term',
                'Term': term,
                'Count': count
            })
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print("\nTotal Function Words per Category:")
    for category in categories:
        total_func_words = sum(func_word_counts_by_category.get(category, {}).values())
        print(f" {reshape_arabic_text(category):<20}: {total_func_words:>6}")
    print("\nTotal Punctuation Marks per Category:")
    for category in categories:
        total_punct = sum(punct_counts_by_category.get(category, {}).values())
        print(f" {reshape_arabic_text(category):<20}: {total_punct:>6}")
    lexical_df = pd.DataFrame(lexical_results)
    lexical_df.to_csv("lexical_analysis_comparison.csv", index=False, encoding="utf-8-sig")
    print(f"\nLexical analysis comparison saved to 'lexical_analysis_comparison.csv'")


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================

def word_counts(text, short_length=3):
    if pd.isna(text) or not isinstance(text, str) or len(text.strip()) == 0:
        return 0, 0, 0
    tokens = word_tokenize(text)
    total_words = len(tokens)
    tokens_no_stop = [word for word in tokens if word not in arabic_stopwords]
    short_count = sum(1 for word in tokens_no_stop if len(word) <= short_length)
    stopwords_count = len(tokens) - len(tokens_no_stop)
    return total_words, stopwords_count, short_count


def compute_foreign(text):
    if pd.isna(text) or not isinstance(text, str) or len(text.strip()) == 0:
        return 0, 0.0
    letters = [c for c in text if c.isalpha()]
    total_letters = len(letters)
    foreign_count = sum(1 for c in letters if not arabic_pattern.match(c))
    foreign_ratio = foreign_count / total_letters if total_letters > 0 else 0.0
    return foreign_count, foreign_ratio


def count_active_voice_fast(text):
    """Fast version - nlp loaded globally"""
    if not isinstance(text, str) or not text.strip():
        return 0
    try:
        doc = nlp(text)
        if not doc.sentences:
            return 0
        active = 0
        for sent in doc.sentences:
            if len(sent.words) < 2:
                continue
            deprels = [w.deprel for w in sent.words if w.deprel]
            if any(d == 'nsubj' for d in deprels):
                active += 1
        return active
    except:
        return 0


def extract_short_words_ratio(
    df: pd.DataFrame,
    text_column: str = 'processed_text',
    short_length: int = 3,
    remove_stopwords: bool = True
) -> pd.DataFrame:
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame!")
    def short_words_ratio(text: str) -> float:
        if pd.isna(text) or not isinstance(text, str) or len(text.strip()) == 0:
            return 0.0
        try:
            tokens = word_tokenize(text)
            tokens = [word for word in tokens if word.isalpha()]
            if remove_stopwords:
                tokens = [word for word in tokens if word not in arabic_stopwords]
            N = len(tokens)
            if N == 0:
                return 0.0
            short_count = sum(1 for word in tokens if len(word) <= short_length)
            return short_count / N
        except Exception as e:
            print(f"Warning: Unexpected error while processing text: {e}")
            return 0.0
    new_column = f'short_words_ratio_{short_length}_rmstop_{remove_stopwords}'
    df[new_column] = df[text_column].apply(short_words_ratio)
    print("\n" + "=" * 80)
    print(f"Feature Extraction Completed: Short Words Ratio (length <= {short_length})")
    print("=" * 80)
    print(f"New column added: '{new_column}'")
    if 'category' in df.columns:
        print("\nCategory-based Statistics (Mean/Std):")
        stats = df.groupby('category')[new_column].agg(['mean', 'std']).round(3)
        print(stats)
    print("\n" + "=" * 80)
    print("Short Words Ratio Feature computation completed successfully.")
    print("=" * 80)
    return df


# ==============================================================================
# REDUNDANCY & DATA SPLIT
# ==============================================================================

def split_into_sentences(text):
    pattern = r'[.!?؛،:۔()\[\]"'']+'
    sentences = re.split(pattern, text.strip())
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]


def redundancy_score_sentences(text, tfidf_vector, threshold=0.1):
    tfidf_array = tfidf_vector.toarray()[0]
    feature_names = vectorizer.get_feature_names_out()
    total_terms = np.count_nonzero(tfidf_array)
    if total_terms == 0:
        return 0.0
    low_tfidf_terms = sum(1 for val, name in zip(tfidf_array, feature_names) if val < threshold)
    score = min(1.0, low_tfidf_terms / total_terms)
    return score


def classify_redundancy(score):
    if score >= 0.7:
        return "Very High Redundancy"
    elif score >= 0.4:
        return "Moderate Redundancy"
    else:
        return "Low Redundancy"


# ==============================================================================
# MODEL DATA SPLIT & BERT EMBEDDINGS
# ==============================================================================

BATCH_SIZE = 16
MAX_LENGTH = 256
USE_GPU = True
OUTPUT_DIR = "Data_Split"
BERT_SAVE_DIR = os.path.join(OUTPUT_DIR, "bert_batches")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BERT_SAVE_DIR, exist_ok=True)

device = torch.device('cuda' if USE_GPU and torch.cuda.is_available() else 'cpu')
model_name = "aubmindlab/bert-base-arabert"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    tokenizer = None
    model = None


def save_bert_embeddings_progressively(texts, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, save_dir=BERT_SAVE_DIR):
    
    if tokenizer is None or model is None:
        print("Skipping BERT embeddings due to model load failure.")
        return
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing Batches"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts,
                           return_tensors="pt",
                           truncation=True,
                           padding=True,
                           max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        np.save(os.path.join(save_dir, f"bert_embeddings_batch_{i//batch_size}.npy"), batch_embeddings)
        if USE_GPU:
            torch.cuda.empty_cache()

