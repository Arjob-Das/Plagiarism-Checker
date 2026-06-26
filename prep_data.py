import pandas as pd
from sklearn.model_selection import train_test_split
import os

def prepare_dataset(data_path='data.csv', subset_size=25000, random_state=42):
    print("Loading dataset...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Source data file '{data_path}' not found.")
        
    df = pd.read_csv(data_path)
    
    # Drop rows with missing values
    df.dropna(subset=['source_txt', 'plagiarism_txt', 'label'], inplace=True)
    
    # Convert label to integer
    df['label'] = df['label'].astype(int)
    
    # Filter valid labels (0 and 1)
    df = df[df['label'].isin([0, 1])]
    
    print(f"Total available rows: {len(df)}")
    
    # Balance the dataset (subset_size / 2 for each class)
    half_size = subset_size // 2
    df_0 = df[df['label'] == 0]
    df_1 = df[df['label'] == 1]
    
    if len(df_0) < half_size or len(df_1) < half_size:
        min_size = min(len(df_0), len(df_1))
        print(f"Warning: Not enough samples for balanced subset of {subset_size}. Using {min_size * 2} samples instead.")
        half_size = min_size
        subset_size = min_size * 2
        
    df_0_sampled = df_0.sample(n=half_size, random_state=random_state)
    df_1_sampled = df_1.sample(n=half_size, random_state=random_state)
    
    balanced_df = pd.concat([df_0_sampled, df_1_sampled]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"Balanced subset size: {len(balanced_df)}")
    print(balanced_df['label'].value_counts())
    
    # Split into train (80%), validation (10%), test (10%)
    train_df, temp_df = train_test_split(balanced_df, test_size=0.2, random_state=random_state, stratify=balanced_df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=random_state, stratify=temp_df['label'])
    
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    
    # Save splits
    train_df.to_csv('train.csv', index=False)
    val_df.to_csv('val.csv', index=False)
    test_df.to_csv('test.csv', index=False)
    print("Splits saved to train.csv, val.csv, and test.csv")

if __name__ == '__main__':
    prepare_dataset()
