import pandas as pd
import numpy as np
import pickle
import json
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from prep_data import prepare_dataset
from siamese_model import build_siamese_model

def train_siamese_network():
    # 1. Prepare data splits if they don't exist
    if not (os.path.exists('train.csv') and os.path.exists('val.csv') and os.path.exists('test.csv')):
        prepare_dataset(subset_size=25000)
        
    print("Loading data splits...")
    train_df = pd.read_csv('train.csv')
    val_df = pd.read_csv('val.csv')
    
    # Fill any empty values
    train_df.fillna('', inplace=True)
    val_df.fillna('', inplace=True)
    
    # 2. Tokenizer setup
    print("Fitting tokenizer...")
    tokenizer = Tokenizer(oov_token='<OOV>')
    # Fit on all texts combined
    all_texts = pd.concat([train_df['source_txt'], train_df['plagiarism_txt']]).astype(str).tolist()
    tokenizer.fit_on_texts(all_texts)
    
    vocab_size = len(tokenizer.word_index) + 1
    max_len = 100  # standard fixed sequence length
    
    print(f"Vocabulary size: {vocab_size}")
    
    # Save tokenizer immediately
    with open('tokenizer.pkl', 'wb') as file:
        pickle.dump(tokenizer, file)
    print("Saved tokenizer.pkl")
    
    # Save metadata configuration
    metadata = {'max_len': max_len}
    with open('model_metadata.json', 'w') as file:
        json.dump(metadata, file)
    print("Saved model_metadata.json")
    
    # 3. Convert texts to sequences and pad them
    print("Converting texts to padded sequences...")
    train_a = pad_sequences(tokenizer.texts_to_sequences(train_df['source_txt'].astype(str)), maxlen=max_len)
    train_b = pad_sequences(tokenizer.texts_to_sequences(train_df['plagiarism_txt'].astype(str)), maxlen=max_len)
    train_labels = train_df['label'].values
    
    val_a = pad_sequences(tokenizer.texts_to_sequences(val_df['source_txt'].astype(str)), maxlen=max_len)
    val_b = pad_sequences(tokenizer.texts_to_sequences(val_df['plagiarism_txt'].astype(str)), maxlen=max_len)
    val_labels = val_df['label'].values
    
    # 4. Build Siamese model
    print("Building model...")
    model = build_siamese_model(vocab_size=vocab_size, embedding_dim=100, max_len=max_len)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    # 5. Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath='model_pdf.keras', monitor='val_loss', save_best_only=True, verbose=1)
    ]
    
    # 6. Train the model
    print("Starting training...")
    history = model.fit(
        x=[train_a, train_b],
        y=train_labels,
        validation_data=([val_a, val_b], val_labels),
        epochs=8,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )
    
    print("Training complete.")
    
    # Save as .h5 version too for backward compatibility
    try:
        model.save('model_pdf.h5')
        print("Saved model_pdf.h5 for compatibility")
    except Exception as e:
        print(f"Warning: Could not save model in legacy H5 format: {e}")
        
    # 7. Quantization/Clone (following user's exact quantization steps)
    print("Saving quantized model versions...")
    try:
        model.save('quantized_model_pdf.keras')
        model.save('quantized_model_pdf.h5')
        print("Saved quantized model versions: quantized_model_pdf.keras, quantized_model_pdf.h5")
    except Exception as e:
        print(f"Error saving quantized models: {e}")

if __name__ == '__main__':
    train_siamese_network()
