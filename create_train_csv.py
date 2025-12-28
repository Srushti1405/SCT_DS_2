import seaborn as sns
import pandas as pd

# Load titanic dataset from seaborn
df = sns.load_dataset("titanic")

# Convert to CSV (train-style dataset)
df.to_csv("titanic.csv", index=False)

print("titanic.csv created successfully!")
print(df.head())
