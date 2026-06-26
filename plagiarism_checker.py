import fitz  # PyMuPDF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import sys
import os
import nltk
import pickle
import json
import argparse

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def preprocess(text):
    """Preprocesses the text by removing non-alphabetic characters, stop words, and stemming the words."""
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    try:
        tokens = word_tokenize(text.lower())
    except Exception:
        tokens = text.lower().split()
        
    tokens = [stemmer.stem(
        token) for token in tokens if token.isalpha() and token not in stop_words]

    return ' '.join(tokens)


def calculate_similarity(text1, text2):
    """Calculates the cosine similarity score between two texts using TF-IDF."""
    texts = [preprocess(text1), preprocess(text2)]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(texts)

    return cosine_similarity(tfidf)[0][1]


def main():
    parser = argparse.ArgumentParser(description="Plagiarism Checker CLI tool")
    parser.add_argument('--input', type=str, default='file2.txt', help='Path to the query text file to check')
    parser.add_argument('--inputs-dir', type=str, default='inputs', help='Directory containing reference PDF files to check against')
    args = parser.parse_args()

    source = []
    plag = []
    input_folder = args.inputs_dir
    filenames = []
    tfidf_scores = []
    dl_scores = []
    scores2 = []
    inp = args.input
    max_tfidf_score = 0
    max_dl_score = 0
    max_tfidf_file = ""
    max_dl_file = ""
    
    if not os.path.isfile(inp):
        print(f"Input file '{inp}' not found.")
        sys.exit(1)

    with open(inp, 'r', encoding='utf-8') as f:
        input_text = f.read()

    plagiarized = False

    # Try loading Siamese model and tokenizer
    model = None
    tokenizer = None
    max_len = 100
    
    print("Checking for trained Siamese model...")
    if os.path.exists('tokenizer.pkl'):
        try:
            with open('tokenizer.pkl', 'rb') as file:
                tokenizer = pickle.load(file)
            
            # Load metadata
            if os.path.exists('model_metadata.json'):
                with open('model_metadata.json', 'r') as file:
                    max_len = json.load(file).get('max_len', 100)
            
            # Load Keras model
            from tensorflow.keras.models import load_model
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            
            model_path = 'model_pdf.keras'
            if not os.path.exists(model_path):
                model_path = 'model_pdf.h5'
            if os.path.exists(model_path):
                model = load_model(model_path, safe_mode=False)
                print("Siamese BiLSTM model loaded successfully.")
            else:
                print("Siamese BiLSTM model file not found. Running TF-IDF only.")
        except Exception as e:
            print(f"Warning: Could not load deep learning model/tokenizer: {e}. Running TF-IDF only.")
    else:
        print("tokenizer.pkl not found. Running TF-IDF only.")

    for file in os.listdir(input_folder):
        if file == inp:
            continue  # Skip the file to check against others

        file_path = os.path.join(input_folder, file)
        if os.path.isfile(file_path):
            src_text = extract_text_from_pdf(file_path)
            source.append(src_text)
            plag.append(input_text)

            # 1. Calculate TF-IDF Cosine Similarity
            similarity_score = calculate_similarity(src_text, input_text)
            scores2.append(similarity_score)
            
            # 2. Calculate Siamese Deep Learning Similarity if model is available
            dl_score = 0.0
            if model is not None and tokenizer is not None:
                try:
                    query_seq = pad_sequences(tokenizer.texts_to_sequences([input_text]), maxlen=max_len)
                    # Take first chunk of source text
                    doc_seq = pad_sequences(tokenizer.texts_to_sequences([src_text[:2000]]), maxlen=max_len)
                    dl_score = model.predict([query_seq, doc_seq], verbose=0)[0][0]
                except Exception as e:
                    print(f"Error checking {file} with deep learning model: {e}")
            
            filenames.append(file)
            tfidf_scores.append(f"{similarity_score*100:0.2f}%")
            dl_scores.append(f"{dl_score*100:0.2f}%" if model is not None else "N/A")
            
            if similarity_score > 0.5 or dl_score > 0.5:
                plagiarized = True
                
            if similarity_score > max_tfidf_score:
                max_tfidf_score = similarity_score
                max_tfidf_file = file
                
            if dl_score > max_dl_score:
                max_dl_score = dl_score
                max_dl_file = file

    if filenames:
        df_results = pd.DataFrame({
            'File Name': filenames,
            'TF-IDF Similarity': tfidf_scores,
            'Siamese Similarity': dl_scores
        })
        print("\n\n")
        print("="*65)
        print("                PLAGIARISM CHECK RESULTS")
        print("="*65)
        print(df_results.to_string(index=False))
        print("="*65)
        
        print(f"\nMaximum TF-IDF Plagiarism: '{inp}' matches '{max_tfidf_file}' (Score: {max_tfidf_score*100:.2f}%)")
        if model is not None:
            print(f"Maximum Siamese Plagiarism: '{inp}' matches '{max_dl_file}' (Score: {max_dl_score*100:.2f}%)")
            
    labels = [1 if x >= 0.4 else 0 for x in scores2]
    df = pd.DataFrame({
        'source_txt': source,
        'plagiarism_txt': plag,
        'label': labels
    })

    df.to_csv(f"{inp.split('.')[0]}_plag.csv")
    print(f"\nDataset saved to {inp.split('.')[0]}_plag.csv")


if __name__ == '__main__':
    main()
