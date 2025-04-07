import pandas as pd
df = pd.read_csv('data.csv')
df = df.drop(['plagiarism_txt', 'label'], axis=1)
print(df.head())
df.to_csv('dataset2.csv')