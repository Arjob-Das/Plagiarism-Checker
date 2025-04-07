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
import pandas as pd
from tqdm import tqdm

# nltk.download('punkt')
# nltk.download('stopwords')


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

    tokens = word_tokenize(text.lower())
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
    input_folder = "inputs"
    filenames = []
    scores = []
    inp = "file3.txt"
    maxscore = 0
    maxfile = ""
    if not os.path.isfile(inp):
        print(f"Input file '{inp}' not found.")
        sys.exit(1)

    with open(inp, 'r', encoding='utf-8') as f:
        input_text = f.read()

    plagiarized = False

    for file in os.listdir(input_folder):
        if file == inp:
            continue  # Skip the file to check against others

        file_path = os.path.join(input_folder, file)
        if os.path.isfile(file_path):
            src_text = extract_text_from_pdf(file_path)

            similarity_score = calculate_similarity(src_text, input_text)

            if similarity_score == 1:
                print("Exact Match!")
                sys.exit(0)

            if similarity_score > 0.4:
                filenames.append(file)
                scores.append(f"{similarity_score*100:0.2f}%")
                plagiarized = True
                if similarity_score > maxscore:
                    maxscore = similarity_score
                    maxfile = file
    if plagiarized:
        df = pd.DataFrame({
            'File Name': filenames,
            'Score': scores
        })
        print("\n\n")
        print(df)
        print(
            f"\nMaximum Plagiarism detected : '{inp}' is most similar to '{maxfile}' (Score: {maxscore*100:.2f}%)\n")
    if not plagiarized:
        print(
            f"No plagiarism detected for '{inp}' against files in '{input_folder}'.")


if __name__ == '__main__':
    main()
