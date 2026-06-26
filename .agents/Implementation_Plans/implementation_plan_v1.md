# Implementation Plan: Enhancing Plagiarism Detection System (v1)

This plan outlines the enhancements to the model architecture, training pipeline, inference logic, and user functionality of the Plagiarism Detection System.

## Proposed Changes

### 1. Data & Dependencies Component

We will create a virtual environment, install the required packages (including TensorFlow, scikit-learn, PyMuPDF, etc.), and implement consistent text preprocessing.

- **[MODIFY] [requirements.txt](file:///f:/Self_Study/Plagiarism-Checker/requirements.txt)**: Update dependencies to ensure compatibility.
- **[NEW] [prep_data.py](file:///f:/Self_Study/Plagiarism-Checker/prep_data.py)**: Dataset utility script to clean text, tokenize, and split into train/val/test sets.

### 2. Model & Training Component

- **[NEW] [siamese_model.py](file:///f:/Self_Study/Plagiarism-Checker/siamese_model.py)**: Defines the Siamese BiLSTM neural network architecture using Keras.
- **[NEW] [train_siamese.py](file:///f:/Self_Study/Plagiarism-Checker/train_siamese.py)**: Training script that trains the Siamese BiLSTM model on a 25,000-sample subset of `data.csv`.

### 3. Verification & Inference Component

- **[MODIFY] [plagiarism_checker.py](file:///f:/Self_Study/Plagiarism-Checker/plagiarism_checker.py)**: Enhanced to run TF-IDF and deep learning Siamese similarity checks on PDF inputs.
- **[NEW] [eval_report.py](file:///f:/Self_Study/Plagiarism-Checker/eval_report.py)**: Benchmark script that outputs baseline vs. current statistics (F1-score, precision, recall, accuracy).
- **[NEW] [predict_widget.py](file:///f:/Self_Study/Plagiarism-Checker/predict_widget.py)**: Interactive widget that uses the Siamese model to perform queries against document databases.

## Verification Plan

### Automated Tests
```bash
.venv/Scripts/python eval_report.py
.venv/Scripts/python plagiarism_checker.py
```
