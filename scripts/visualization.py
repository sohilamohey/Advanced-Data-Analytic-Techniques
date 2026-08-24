# visualization.py
# Functions for generating plots and visualizations.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bidi.algorithm import get_display
import arabic_reshaper
from wordcloud import WordCloud
from matplotlib.font_manager import FontProperties, findfont
from collections import Counter
from nltk.util import ngrams
import os
import gc
import plotly.express as px
from sklearn.metrics import confusion_matrix

# ==============================================================================
# SETUP AND HELPER FUNCTIONS
# ==============================================================================

def setup_visualization_environment():
    """Configures the plotting environment for Arabic text visualization."""
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.unicode_minus"] = False

    arabic_fonts = ["Arial", "Amiri", "DejaVu Sans", "Tahoma"]
    for font in arabic_fonts:
        try:
            font_path = findfont(FontProperties(family=font))
            if os.path.exists(font_path):
                plt.rcParams["font.family"] = font
                print(f"Using font: {font}")
                return font_path
        except:
            continue
    
    print("Warning: No suitable Arabic font found.")
    plt.rcParams["font.family"] = "sans-serif"
    return None


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


def reshape_arabic_text(text):
    """Helper function to reshape Arabic text for proper display."""
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)


# ==============================================================================
# WORD CLOUD VISUALIZATION
# ==============================================================================

def plot_wordclouds_simple(df: pd.DataFrame, font_path: str | None, max_words: int = 30):
    """
    Generates smaller word clouds for each category with memory optimization.
    """
    if "category_name" not in df.columns or "processed_text" not in df.columns:
        print("Error: Required columns 'category_name' or 'processed_text' not found")
        return

    if not font_path:
        print("Warning: No Arabic font path available. Skipping word clouds to avoid rendering issues.")
        return

    categories = df["category_name"].unique()
    
    print(f"Generating word clouds for {len(categories)} categories...")
    
    for category in categories:
        try:
            # Clear memory before each plot
            gc.collect()
            
            category_texts = df[df["category_name"] == category]["processed_text"].dropna()
            combined_text = " ".join(category_texts)
            
            if not combined_text.strip():
                print(f"No text data for category: {category}")
                continue
            
            # Limit text size to prevent memory issues
            if len(combined_text) > 10000:
                combined_text = combined_text[:10000]
            
            # Create figure with smaller size 
            plt.figure(figsize=(6, 3))
            
            # Generate word cloud with smaller dimensions
            reshaped_text = arabic_reshaper.reshape(combined_text)
            bidi_text = get_display(reshaped_text)
            
            wordcloud = WordCloud(
                font_path=font_path,
                width=400,  
                height=200, 
                background_color="white",
                colormap="viridis",
                max_words=max_words,
                relative_scaling=0.3,
                random_state=42
            ).generate(bidi_text)
            
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.title(reshape_arabic_text(f"Word Cloud - {category}"), fontsize=10, pad=15)  # Reduced fontsize
            plt.axis("off")
            plt.tight_layout()
            
            # Save and show
            filename = f"wordcloud_{category.replace(' ', '_')}.png"
            plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor='white')
            plt.show()
            plt.close()
            
            print(f"Word cloud saved as '{filename}'")
            
        except MemoryError:
            print(f"Memory error generating word cloud for {category}. Skipping...")
            plt.close()
            continue
        except Exception as e:
            print(f"Error generating word cloud for {category}: {str(e)}")
            plt.close()
            continue


# ==============================================================================
# N-GRAM VISUALIZATION
# ==============================================================================

def plot_top_ngrams_combined(df: pd.DataFrame, top_n: int = 8):
    """
    Generates bar plots of the top n-grams for each category with all three types together.
    """
    if "category_name" not in df.columns or "processed_text" not in df.columns:
        print("Error: Required columns 'category_name' or 'processed_text' not found")
        return

    categories = df["category_name"].unique()
    
    print(f"Generating n-gram visualizations for {len(categories)} categories...")
    
    for category in categories:
        try:
            # Clear memory before each plot
            gc.collect()
            
            category_texts = df[df["category_name"] == category]["processed_text"].dropna()
            
            if category_texts.empty:
                print(f"No data for category: {category}")
                continue
            
            # Extract tokens for this category
            all_tokens = []
            for text in category_texts:
                if isinstance(text, str) and text.strip():
                    tokens = text.split()
                    # Limit tokens to prevent memory issues
                    if len(tokens) > 1000:
                        tokens = tokens[:1000]
                    all_tokens.extend(tokens)
            
            # Create figure with 3 subplots side by side
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f"Top N-Grams Analysis - {category}", fontsize=16, y=0.95)
            
            ngram_names = ["Unigrams", "Bigrams", "Trigrams"]
            has_data = False
            
            for n, ax in enumerate(axes, 1):
                try:
                    if len(all_tokens) >= n:
                        # Generate n-grams
                        ngram_counts = Counter(ngrams(all_tokens, n))
                        top_ngrams = ngram_counts.most_common(top_n)
                        
                        if top_ngrams:
                            has_data = True
                            ngrams_list, counts = zip(*top_ngrams)
                            
                            # Prepare labels
                            ngram_labels = []
                            for ng in ngrams_list:
                                arabic_text = " ".join(ng)
                                ngram_labels.append(reshape_arabic_text(arabic_text))
                            
                            # Create horizontal bar plot
                            y_pos = np.arange(len(ngram_labels))
                            bars = ax.barh(y_pos, counts, color=plt.cm.Set3(np.linspace(0, 1, len(ngram_labels))))
                            
                            ax.set_yticks(y_pos)
                            ax.set_yticklabels(ngram_labels, fontsize=9)
                            ax.invert_yaxis()
                            
                            ax.set_title(f"Top {top_n} {ngram_names[n-1]}", fontsize=12)
                            ax.set_xlabel("Frequency", fontsize=10)
                            
                            # Add value labels
                            for i, (bar, count) in enumerate(zip(bars, counts)):
                                width = bar.get_width()
                                ax.text(width + 0.1, i, str(count), va='center', fontsize=8)
                            
                            ax.grid(axis='x', alpha=0.3)
                            ax.tick_params(axis="x", labelsize=8)
                            
                        else:
                            ax.text(0.5, 0.5, reshape_arabic_text("لا توجد بيانات"), 
                                   ha="center", va="center", fontsize=12, transform=ax.transAxes)
                            ax.set_title(f"{ngram_names[n-1]}", fontsize=12)
                    else:
                        ax.text(0.5, 0.5, reshape_arabic_text("بيانات غير كافية"), 
                               ha="center", va="center", fontsize=12, transform=ax.transAxes)
                        ax.set_title(f"{ngram_names[n-1]}", fontsize=12)
                    
                    ax.axis('on')
                    
                except MemoryError:
                    print(f"Memory error generating {n}-grams for {category}. Skipping...")
                    ax.text(0.5, 0.5, "Memory Error", ha="center", va="center", transform=ax.transAxes)
                    continue
                except Exception as e:
                    print(f"Error generating {n}-grams for {category}: {str(e)}")
                    ax.text(0.5, 0.5, "Error", ha="center", va="center", transform=ax.transAxes)
                    continue
            
            if has_data:
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                filename = f"ngrams_combined_{category.replace(' ', '_')}.png"
                plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor='white')
                plt.show()
                plt.close()
                print(f"N-gram plot saved as '{filename}'")
            else:
                plt.close(fig)
                print(f"No sufficient data for n-grams in category: {category}")
                    
        except Exception as e:
            print(f"Error processing category {category}: {str(e)}")
            continue


# ==============================================================================
# SUMMARY STATISTICS VISUALIZATION
# ==============================================================================

def visualize_summary_statistics(df: pd.DataFrame):
    """Creates visualizations for the summary statistics from lexical analysis."""
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS FOR SUMMARY STATISTICS")
    print("="*80)
    
    categories = sorted(df['category_name'].unique())
    all_func_words = set()
    func_word_counts_by_category = {}
    
    for category in categories:
        category_texts = df[df['category_name'] == category]['text']
        func_word_counts = analyze_function_words(category_texts)
        func_word_counts_by_category[category] = func_word_counts
        all_func_words.update(func_word_counts.keys())
    
    # 2. Punctuation Analysis
    all_punctuations = set()
    punct_counts_by_category = {}
    
    for category in categories:
        category_texts = df[df['category_name'] == category]['text']
        punct_counts = analyze_punctuation(category_texts)
        punct_counts_by_category[category] = punct_counts
        all_punctuations.update(punct_counts.keys())
    
    # Prepare data for visualization
    summary_data = []
    
    for category in categories:
        # Total function words
        total_func_words = sum(func_word_counts_by_category.get(category, {}).values())
        
        # Total punctuation marks
        total_punct = sum(punct_counts_by_category.get(category, {}).values())
        
        # Text length for normalization
        total_text_length = sum(len(str(text)) for text in df[df['category_name'] == category]['text'])
        
        summary_data.append({
            'Category': category,
            'Total_Function_Words': total_func_words,
            'Total_Punctuation': total_punct,
            'Function_Words_per_1000_chars': (total_func_words / total_text_length * 1000) if total_text_length > 0 else 0,
            'Punctuation_per_1000_chars': (total_punct / total_text_length * 1000) if total_text_length > 0 else 0
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create visualizations - Only the 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Lexical Analysis - Summary Statistics Visualization', fontsize=16, fontweight='bold')
    
    # 1. Total Function Words by Category
    axes[0, 0].bar(range(len(categories)), summary_df['Total_Function_Words'], 
                   color=plt.cm.Set3(np.linspace(0, 1, len(categories))))
    axes[0, 0].set_title('Total Function Words by Category', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Categories')
    axes[0, 0].set_ylabel('Total Function Words')
    axes[0, 0].set_xticks(range(len(categories)))
    axes[0, 0].set_xticklabels([reshape_arabic_text(cat) for cat in categories], rotation=45, ha='right')
    
    # Add value labels on bars
    for i, v in enumerate(summary_df['Total_Function_Words']):
        axes[0, 0].text(i, v + max(summary_df['Total_Function_Words']) * 0.01, 
                       str(int(v)), ha='center', va='bottom', fontweight='bold')
    
    # 2. Total Punctuation by Category
    axes[0, 1].bar(range(len(categories)), summary_df['Total_Punctuation'], 
                   color=plt.cm.Set3(np.linspace(0.2, 0.8, len(categories))))
    axes[0, 1].set_title('Total Punctuation Marks by Category', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Categories')
    axes[0, 1].set_ylabel('Total Punctuation Marks')
    axes[0, 1].set_xticks(range(len(categories)))
    axes[0, 1].set_xticklabels([reshape_arabic_text(cat) for cat in categories], rotation=45, ha='right')
    
    # Add value labels on bars
    for i, v in enumerate(summary_df['Total_Punctuation']):
        axes[0, 1].text(i, v + max(summary['Total_Punctuation']) * 0.01, 
                       str(int(v)), ha='center', va='bottom', fontweight='bold')
    
    # 3. Function Words per 1000 Characters (Normalized)
    axes[1, 0].bar(range(len(categories)), summary_df['Function_Words_per_1000_chars'], 
                   color=plt.cm.Pastel1(np.linspace(0, 1, len(categories))))
    axes[1, 0].set_title('Function Words Density (per 1000 characters)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Categories')
    axes[1, 0].set_ylabel('Function Words per 1000 chars')
    axes[1, 0].set_xticks(range(len(categories)))
    axes[1, 0].set_xticklabels([reshape_arabic_text(cat) for cat in categories], rotation=45, ha='right')
    
    # Add value labels on bars
    for i, v in enumerate(summary_df['Function_Words_per_1000_chars']):
        axes[1, 0].text(i, v + max(summary_df['Function_Words_per_1000_chars']) * 0.01, 
                       f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Punctuation per 1000 Characters (Normalized)
    axes[1, 1].bar(range(len(categories)), summary_df['Punctuation_per_1000_chars'], 
                   color=plt.cm.Pastel2(np.linspace(0, 1, len(categories))))
    axes[1, 1].set_title('Punctuation Density (per 1000 characters)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Categories')
    axes[1, 1].set_ylabel('Punctuation per 1000 chars')
    axes[1, 1].set_xticks(range(len(categories)))
    axes[1, 1].set_xticklabels([reshape_arabic_text(cat) for cat in categories], rotation=45, ha='right')
    
    # Add value labels on bars
    for i, v in enumerate(summary_df['Punctuation_per_1000_chars']):
        axes[1, 1].text(i, v + max(summary_df['Punctuation_per_1000_chars']) * 0.01, 
                       f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('lexical_analysis_summary.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Print summary table
    print("\nSUMMARY STATISTICS TABLE:")
    print("="*80)
    print(f"{'Category':<25} {'Func Words':<12} {'Punctuation':<12} {'Func/1K chars':<15} {'Punct/1K chars':<15}")
    print("-"*80)
    
    for _, row in summary_df.iterrows():
        print(f"{reshape_arabic_text(row['Category']):<25} {row['Total_Function_Words']:<12} {row['Total_Punctuation']:<12} {row['Function_Words_per_1000_chars']:<15.2f} {row['Punctuation_per_1000_chars']:<15.2f}")
    
    print(f"\nVisualization saved as: 'lexical_analysis_summary.png'")

# ==============================================================================
# CONFUSION MATRIX VISUALIZATION
# ==============================================================================

def save_confusion_matrix_plot(y_true, y_pred, model_name, embedding_name):
    """
    Generate and save confusion matrix as both interactive HTML (Plotly) and PNG (Matplotlib).
    Always uses black text for readability.
    """
    # Base directory (adjust if needed)
    BASE_DIR = globals().get("BASE_DIR", "/content/drive/My Drive/Advanced Data Analytic Techniques")
    # Ensure save directory exists
    cm_dir = os.path.join(BASE_DIR, "confusion_matrices", str(model_name))
    os.makedirs(cm_dir, exist_ok=True)

    # File paths
    base_fname = f"CM_{model_name}_{embedding_name}"
    html_path = os.path.join(cm_dir, base_fname + ".html")
    png_path = os.path.join(cm_dir, base_fname + ".png")

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    labels_sorted = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))
    x_labels = [str(l) for l in labels_sorted]
    y_labels = [str(l) for l in labels_sorted]

    # -------- Plotly interactive (HTML) --------
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=x_labels,
        y=y_labels,
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 14, "color": "black"},
        hoverongaps=False
    ))
    fig.update_layout(
        title=f'Confusion Matrix — {model_name} ({embedding_name})',
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        height=520,
        width=560,
        plot_bgcolor='white',
        font=dict(size=14)
    )
    fig.write_html(html_path, include_plotlyjs='cdn', full_html=True)

    # -------- PNG via Matplotlib --------
    plt.figure(figsize=(5.6, 5.2), dpi=160)
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'{model_name} ({embedding_name})')
    plt.colorbar()
    tick_marks = np.arange(len(labels_sorted))
    plt.xticks(tick_marks, x_labels, rotation=45, ha="right")
    plt.yticks(tick_marks, y_labels)
    # All numbers in black
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="black",
                     fontsize=10)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()

    # -------- Show interactive only --------
    print(f"\nConfusion Matrix: {model_name} ({embedding_name})")
    fig.show()
    return cm

# ==============================================================================
# MODEL PERFORMANCE VISUALIZATIONS (Plotly)
# ==============================================================================

def create_model_scatter(results: pd.DataFrame):
    """Create scatter plot: Precision vs Recall with Accuracy as bubble size."""
    results_sorted = results.sort_values("Accuracy", ascending=False)
    
    fig = px.scatter(
        results_sorted,
        x="Precision", y="Recall",
        size="Accuracy",
        color="Model",
        text="Embedding",
        hover_data=["F1-Score", "ROC-AUC"],
        title="Model Performance Overview (Precision vs Recall)",
        template="plotly_white"
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(height=600, width=900)
    return fig


def create_top10_bar(results: pd.DataFrame):
    """Create horizontal bar chart for top 10 models by Accuracy."""
    results_sorted = results.sort_values("Accuracy", ascending=False)
    top10 = results_sorted.head(10)
    
    fig = px.bar(
        top10,
        x="Accuracy", y="Model",
        color="Embedding",
        orientation="h",
        text=top10["Accuracy"].apply(lambda v: f"{v*100:.2f}%"),
        title="Top 10 Models by Accuracy",
        template="simple_white"
    )
    fig.update_traces(textfont_size=13, textposition="outside")
    fig.update_layout(height=550, width=1000)
    return fig


def create_radar_chart(results: pd.DataFrame):
    """Create radar chart for normalized performance comparison."""
    norm_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    results_norm = results.copy()
    
    for c in norm_cols:
        col_min = results_norm[c].min()
        col_max = results_norm[c].max()
        if col_max > col_min:
            results_norm[c] = (results_norm[c] - col_min) / (col_max - col_min)
        else:
            results_norm[c] = 0
    
    fig = go.Figure()
    for _, row in results_norm.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row[c] for c in norm_cols],
            theta=norm_cols,
            fill='toself',
            name=f"{row['Model']} ({row['Embedding']})"
        ))
    
    fig.update_layout(
        title="Radar Chart: Model Comparison Across Metrics",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=700,
        width=900
    )
    return fig


def create_heatmap(results: pd.DataFrame):
    """Create heatmap for normalized performance summary."""
    norm_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    results_norm = results.copy()
    
    for c in norm_cols:
        col_min = results_norm[c].min()
        col_max = results_norm[c].max()
        if col_max > col_min:
            results_norm[c] = (results_norm[c] - col_min) / (col_max - col_min)
        else:
            results_norm[c] = 0
    
    fig = px.imshow(
        results_norm[norm_cols],
        labels=dict(x="Metric", y="Model (Embedding)", color="Score"),
        x=norm_cols,
        y=[f"{m} ({e})" for m, e in zip(results['Model'], results['Embedding'])],
        color_continuous_scale="Blues",
        title="Model Comparison Heatmap (Normalized)",
        text_auto=".2f",
        aspect="auto"
    )
    fig.update_layout(height=650, width=1100)
    return fig


def save_model_visualizations(figures: dict, base_dir: str = "/content/drive/My Drive/Advanced Data Analytic Techniques"):
    """Save all model performance figures as HTML to the specified directory."""
    results_dir = os.path.join(base_dir, "results_visuals")
    os.makedirs(results_dir, exist_ok=True)
    
    for name, fig in figures.items():
        path = os.path.join(results_dir, f"{name}.html")
        fig.write_html(path)
        print(f"Saved: {name}.html")
    
    print(f"\nAll model visualizations saved to:\n   {results_dir}")
