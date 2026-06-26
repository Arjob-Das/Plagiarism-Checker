# Plagiarism Detection System

This project implements a plagiarism detection system that checks semantic and textual similarity between a query text and a database of documents. It integrates traditional NLP techniques (TF-IDF + Cosine Similarity) and advanced Deep Learning models (Siamese Bidirectional LSTM) to offer a highly robust similarity assessment.

## Features

- **Text Extraction from PDF:** Extracts raw text from PDF files using `PyMuPDF` (`fitz`).
- **Text Preprocessing:** Tokenizes, filters stopwords, and applies Porter stemming for TF-IDF calculations.
- **Side-by-Side Plagiarism Check (CLI):** Compares a query text file against all documents in a directory and outputs similarity scores for both TF-IDF and Siamese BiLSTM models.
- **Siamese BiLSTM Deep Learning Model:** A neural network built with Keras that processes document pairs through shared Embedding and Bidirectional LSTM layers, computing similarity via absolute difference and product merge vectors.
- **Dataset Preparation & Scaling:** Utilities to preprocess, clean, and balance massive textual corpora (like the SNLI dataset) into training, validation, and testing splits.
- **Automated Performance Benchmarking:** Generates performance reports (Accuracy, Precision, Recall, F1-Score, and Confusion Matrices) comparing models.
- **Real-Time Prediction Widget:** An interactive Jupyter notebook widget powered by `ipywidgets` to check custom texts against a PDF database dynamically.

---

## Installation & Setup

1. **Clone the Repository** and navigate to the project directory.

2. **Initialize a Python Virtual Environment:**
   ```bash
   py -m venv .venv
   ```

3. **Activate the Virtual Environment:**
   - **Windows (PowerShell):**
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - **Windows (CMD):**
     ```cmd
     .\.venv\Scripts\activate.bat
     ```
   - **macOS / Linux:**
     ```bash
     source .venv/bin/activate
     ```

4. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage Workflow

### Step 1: Prepare the Splits
Split the larger corpus (`data.csv`) into balanced train, validation, and test datasets:
```bash
python prep_data.py
```
This generates `train.csv`, `val.csv`, and `test.csv`.

### Step 2: Train the Siamese Model
Train the Siamese BiLSTM model on the balanced splits:
```bash
python train_siamese.py
```
This will train the model, save it as `model_pdf.keras`/`model_pdf.h5`, serialize the tokenizer to `tokenizer.pkl`, and generate quantized model weights (`quantized_model_pdf.keras`/`quantized_model_pdf.h5`).

### Step 3: Run the Benchmark Report
Evaluate and compare the performance of the baseline TF-IDF cosine check and the Siamese BiLSTM model on the test set:
```bash
python eval_report.py
```
This prints the comparison table and saves a detailed report to `evaluation_report.md`.

### Step 4: Run Plagiarism Check CLI
Check a document against a folder of reference PDFs:
```bash
python plagiarism_checker.py --input file2.txt --inputs-dir inputs
```
- `--input`: Path to the query text file (defaults to `file2.txt`).
- `--inputs-dir`: Folder containing PDF reference files (defaults to `inputs`).

This outputs a side-by-side comparison table of similarity percentages for all files in the directory.

---

## Project Structure

- **`plagiarism_checker.py`**: Main CLI utility for checking documents side-by-side.
- **`siamese_model.py`**: Model architecture definition for the Siamese BiLSTM network.
- **`train_siamese.py`**: Pipeline script to train and quantize the deep learning model.
- **`prep_data.py`**: Dataset balancing and train/val/test splitting tool.
- **`eval_report.py`**: Performance evaluation and statistical reporter.
- **`predict_widget.py`**: Script powering the interactive prediction widget.
- **`plagiarism.ipynb`**: Original data exploration notebook.
- **`plagarism_tensor_cpu.ipynb`**: Interactive notebook containing the original model training and widget.
- **`requirements.txt`**: List of Python package dependencies.
- **`.gitignore`**: Git path exclusion rules.
- **`.antigravityignore`**: AntiGravity optimization rules.

---

## Model Benchmark Results

On an unseen test set partition of 2,500 sentence pairs, the metrics compare as follows:

| Metric | TF-IDF Cosine Baseline | Siamese BiLSTM Model | Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 60.08% | **75.48%** | **+15.40%** |
| **Precision** | 70.93% | **75.75%** | **+4.82%** |
| **Recall** | 34.16% | **74.96%** | **+40.80%** |
| **F1-Score** | 46.11% | **75.35%** | **+29.24%** |

The Siamese model's deep semantic understanding provides a **+40.80% increase in Recall**, which is crucial for identifying paraphrased plagiarism.
