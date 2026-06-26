import os
import pickle
import json
import fitz  # PyMuPDF
import ipywidgets as widgets
from IPython.display import display
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# 1. Load resources
print("Loading model and tokenizer...")
if not os.path.exists('tokenizer.pkl'):
    raise FileNotFoundError("tokenizer.pkl not found. Please train the model first.")
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)
    
max_len = 100
if os.path.exists('model_metadata.json'):
    with open('model_metadata.json', 'r') as file:
        max_len = json.load(file).get('max_len', 100)
        
model_path = 'model_pdf.keras'
if not os.path.exists(model_path):
    model_path = 'model_pdf.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError("Trained Siamese model not found.")
    
model = load_model(model_path, safe_mode=False)
print("Model loaded successfully.")

# 2. Extract and cache text from PDFs in inputs/
input_folder = "inputs"
reference_docs = {}

print(f"Reading reference documents from '{input_folder}'...")
if os.path.exists(input_folder):
    for file in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file)
        if os.path.isfile(file_path) and file.endswith('.pdf'):
            try:
                # Extract text
                text = ""
                with fitz.open(file_path) as doc:
                    for page in doc:
                        text += page.get_text()
                if text.strip():
                    reference_docs[file] = text
                    print(f"Cached: {file} ({len(text)} chars)")
            except Exception as e:
                print(f"Error reading {file}: {e}")
else:
    print(f"Warning: Folder '{input_folder}' not found.")

# 3. Widget UI setup
text_input = widgets.Text(
    value='Electric Vehicles can mitigate global warming.',
    description='Query Text:',
    placeholder='Type text to check for plagiarism...',
    layout=widgets.Layout(width='600px'),
    disabled=False
)
output_area = widgets.Output()

def predict_plagiarism(new_text):
    if not new_text.strip() or not reference_docs:
        return "No text or reference documents available.", 0.0
        
    scores = {}
    
    # Pre-tokenize the query text once
    query_seq = pad_sequences(tokenizer.texts_to_sequences([new_text]), maxlen=max_len)
    
    # Compare against each cached reference document
    for doc_name, doc_text in reference_docs.items():
        # Pre-tokenize reference text (take first chunk or sample)
        # Since reference text can be long, we take the first 500 words or so, or a relevant window.
        # For simple pairwise classification, we can tokenize the first segment.
        doc_seq = pad_sequences(tokenizer.texts_to_sequences([doc_text[:2000]]), maxlen=max_len)
        
        # Predict similarity score
        pred = model.predict([query_seq, doc_seq], verbose=0)
        scores[doc_name] = pred[0][0]
        
    # Get highest similarity score
    if not scores:
        return "No similarity scores calculated.", 0.0
        
    most_similar_doc = max(scores, key=scores.get)
    max_score = scores[most_similar_doc]
    
    return most_similar_doc, max_score

def on_type(change):
    with output_area:
        output_area.clear_output()
        new_text = change.new
        if not new_text.strip():
            return
            
        doc_name, score = predict_plagiarism(new_text)
        
        if score > 0.5:
            message = f"⚠️ Plagiarism Detected! (Score: {score*100:.2f}%)\nMost similar file: '{doc_name}'"
        else:
            message = f"✅ No Significant Plagiarism Detected. (Highest Sim: {score*100:.2f}% with '{doc_name}')"
        print(message)

# Observer setup
text_input.observe(on_type, names='value')

print("\nInteractive Plagiarism Checker Widget Ready.")
display(text_input, output_area)
# Trigger initial run
on_type(type('Change', (object,), {'new': text_input.value})())
