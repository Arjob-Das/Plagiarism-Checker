# Plagiarism Detection System

This project implements a plagiarism detection system that checks the similarity between a reference text and multiple documents using Natural Language Processing (NLP) techniques and machine learning models. The system extracts text from PDF files, preprocesses it, calculates cosine similarity, and uses a machine learning model (LSTM) to predict the presence of plagiarism.

## Features

- **Text Extraction from PDF:** Extracts raw text from PDF files using `PyMuPDF`.
- **Preprocessing:** Tokenizes, removes stopwords, applies stemming, and cleans text for better feature extraction.
- **Plagiarism Detection:** Uses TF-IDF and cosine similarity to calculate the similarity score between documents.
- **Machine Learning Model:** An LSTM-based deep learning model is trained on labeled data to classify whether the text contains plagiarism.
- **Quantization:** After training, the model is quantized to optimize performance.
- **Real-time Prediction:** Provides real-time plagiarism predictions via a simple user interface built with `ipywidgets`.

## Installation

To run the project, ensure you have the necessary dependencies. You can install them via pip:

```bash
pip install -r requirements.txt
```
## Required Libraries:
fitz (PyMuPDF) for text extraction from PDFs.

sklearn for machine learning tools (TF-IDF, cosine similarity, etc.).

nltk for text preprocessing (tokenization, stemming, stopwords).

tensorflow for building and training the deep learning model.

ipywidgets for creating interactive widgets for real-time prediction.

## Usage
### Step 1: Prepare the Dataset
Place your PDF files in the inputs folder. Create a text file file2.txt containing the reference text to compare against other documents.

### Step 2: Run Plagiarism Check
To check plagiarism in your documents, simply run the main script:

```bash

python plagiarism_check.py
```
The script will:

Extract and preprocess text from the PDFs.

Calculate cosine similarity between the input text and each document.

Display results showing files with high similarity scores and potential plagiarism.

Save the results to a CSV file.

### Step 3: Train the Model (Optional)
You can train the model on a custom dataset by providing labeled data in a CSV format with columns: source_txt, plagiarism_txt, and label.

Run the python_tensor_cpu.ipynb to preprocess data and train the LSTM model:


This will train the model and save it as model_pdf.h5, along with the tokenizer as tokenizer.pkl.

### Step 4: Make Real-time Predictions
You can use the interactive widget for real-time plagiarism detection by using python_tensor_cpu.ipynb 

This will launch an interactive text box where you can input new text and get an immediate plagiarism prediction.

## Files
plagiarism_check.py: Main script for plagiarism detection.

python_tensor_cpu.ipynb: Script to train the LSTM model. Interactive script for real-time plagiarism prediction.

requirements.txt: List of Python dependencies required to run the project.

model_pdf.h5: Pretrained LSTM model.

tokenizer.pkl: Tokenizer for text preprocessing.

## Model Details
The model is built using TensorFlow and Keras. It consists of two LSTM layers followed by a dense layer for binary classification.

It uses binary cross-entropy loss and the Adam optimizer.

The model is quantized after training to optimize for performance.

## Acknowledgments
The plagiarism detection approach is based on NLP techniques using TF-IDF and cosine similarity.

The LSTM model was trained using the tensorflow.keras library.

The interactive real-time prediction interface is powered by ipywidgets.
