import pandas as pd
import numpy as np
import pickle
import json
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading required NLTK datasets...")
    nltk.download('punkt')
    nltk.download('stopwords')

def preprocess_text(text):
    """Identical preprocessing as plagiarism_checker.py"""
    if not isinstance(text, str):
        return ""
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    try:
        tokens = word_tokenize(text.lower())
    except Exception:
        tokens = text.lower().split()
    tokens = [stemmer.stem(token) for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

def evaluate_tfidf(df):
    print("Evaluating TF-IDF Cosine Similarity Baseline on test set...")
    y_true = df['label'].values
    y_pred = []
    
    # We will compute cosine similarity for each pair
    for idx, row in df.iterrows():
        txt1 = preprocess_text(row['source_txt'])
        txt2 = preprocess_text(row['plagiarism_txt'])
        
        if not txt1.strip() or not txt2.strip():
            y_pred.append(0)
            continue
            
        vectorizer = TfidfVectorizer()
        try:
            tfidf = vectorizer.fit_transform([txt1, txt2])
            sim = cosine_similarity(tfidf)[0][1]
        except Exception:
            sim = 0
            
        # Threshold identical to plagiarism_checker.py (0.4)
        y_pred.append(1 if sim >= 0.4 else 0)
        
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        'Confusion_Matrix': confusion_matrix(y_true, y_pred).tolist()
    }

def evaluate_siamese(df):
    print("Evaluating Siamese BiLSTM Model on test set...")
    # Load tokenizer
    if not os.path.exists('tokenizer.pkl'):
        raise FileNotFoundError("tokenizer.pkl not found. Please train the model first.")
    with open('tokenizer.pkl', 'rb') as file:
        tokenizer = pickle.load(file)
        
    # Load metadata
    max_len = 100
    if os.path.exists('model_metadata.json'):
        with open('model_metadata.json', 'r') as file:
            max_len = json.load(file).get('max_len', 100)
            
    # Load model
    model_path = 'model_pdf.keras'
    if not os.path.exists(model_path):
        model_path = 'model_pdf.h5'
    if not os.path.exists(model_path):
        raise FileNotFoundError("Trained Siamese model not found.")
        
    model = load_model(model_path, safe_mode=False)
    
    # Tokenize and pad
    test_a = pad_sequences(tokenizer.texts_to_sequences(df['source_txt'].astype(str)), maxlen=max_len)
    test_b = pad_sequences(tokenizer.texts_to_sequences(df['plagiarism_txt'].astype(str)), maxlen=max_len)
    y_true = df['label'].values
    
    # Predict similarity
    preds = model.predict([test_a, test_b], batch_size=128)
    y_pred = (preds > 0.5).astype(int).flatten()
    
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        'Confusion_Matrix': confusion_matrix(y_true, y_pred).tolist()
    }

def run_evaluation():
    if not os.path.exists('test.csv'):
        print("Test split 'test.csv' not found. Please run prep_data.py or train_siamese.py first.")
        return
        
    test_df = pd.read_csv('test.csv')
    test_df.fillna('', inplace=True)
    
    # Evaluate baseline
    tfidf_metrics = evaluate_tfidf(test_df)
    
    # Evaluate new model
    try:
        siamese_metrics = evaluate_siamese(test_df)
    except Exception as e:
        print(f"Error evaluating Siamese model: {e}")
        siamese_metrics = None
        
    # Print comparison
    print("\n" + "="*50)
    print("           MODEL PERFORMANCE COMPARISON")
    print("="*50)
    
    metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    comparison_data = []
    
    for metric in metrics_list:
        tfidf_val = tfidf_metrics[metric]
        siamese_val = siamese_metrics[metric] if siamese_metrics else 0.0
        comparison_data.append({
            'Metric': metric,
            'TF-IDF Baseline': f"{tfidf_val:.4f}",
            'Siamese BiLSTM (New)': f"{siamese_val:.4f}",
            'Delta': f"{siamese_val - tfidf_val:+.4f}"
        })
        
    report_df = pd.DataFrame(comparison_data)
    print(report_df.to_string(index=False))
    print("="*50)
    
    # Save statistics report
    stats_data = {
        'tfidf_metrics': tfidf_metrics,
        'siamese_metrics': siamese_metrics
    }
    with open('evaluation_results.json', 'w') as file:
        json.dump(stats_data, file, indent=4)
    print("Saved evaluation_results.json")
    
    s_acc = siamese_metrics['Accuracy'] if siamese_metrics else 0.0
    s_prec = siamese_metrics['Precision'] if siamese_metrics else 0.0
    s_rec = siamese_metrics['Recall'] if siamese_metrics else 0.0
    s_f1 = siamese_metrics['F1-Score'] if siamese_metrics else 0.0

    # Generate markdown report
    markdown_report = f"""# Performance Evaluation Report

Comparison of the baseline TF-IDF cosine similarity model vs. the new deep learning Siamese BiLSTM model.

| Metric | TF-IDF Baseline (Cosine Sim >= 0.4) | Siamese BiLSTM Model | Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | {tfidf_metrics['Accuracy']:.4f} | {s_acc:.4f} | {s_acc - tfidf_metrics['Accuracy']:+.4f} |
| **Precision** | {tfidf_metrics['Precision']:.4f} | {s_prec:.4f} | {s_prec - tfidf_metrics['Precision']:+.4f} |
| **Recall** | {tfidf_metrics['Recall']:.4f} | {s_rec:.4f} | {s_rec - tfidf_metrics['Recall']:+.4f} |
| **F1-Score** | {tfidf_metrics['F1-Score']:.4f} | {s_f1:.4f} | {s_f1 - tfidf_metrics['F1-Score']:+.4f} |

### Confusion Matrices

**TF-IDF Baseline:**
- True Negatives: {tfidf_metrics['Confusion_Matrix'][0][0]}
- False Positives: {tfidf_metrics['Confusion_Matrix'][0][1]}
- False Negatives: {tfidf_metrics['Confusion_Matrix'][1][0]}
- True Positives: {tfidf_metrics['Confusion_Matrix'][1][1]}

"""
    if siamese_metrics:
        markdown_report += f"""
**Siamese BiLSTM:**
- True Negatives: {siamese_metrics['Confusion_Matrix'][0][0]}
- False Positives: {siamese_metrics['Confusion_Matrix'][0][1]}
- False Negatives: {siamese_metrics['Confusion_Matrix'][1][0]}
- True Positives: {siamese_metrics['Confusion_Matrix'][1][1]}
"""

    with open('evaluation_report.md', 'w', encoding='utf-8') as file:
        file.write(markdown_report)
    print("Saved evaluation_report.md")

if __name__ == '__main__':
    run_evaluation()
