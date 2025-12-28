import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("titanic.csv")

# View first 5 rows
print(df.head())

# Dataset info
print(df.info())

# Statistical summary
print(df.describe())

# Check missing values
print(df.isnull().sum())

# ===============================
# Data Cleaning (FIXED COLUMN NAMES)
# ===============================

# Fill missing age with mean
df['age'].fillna(df['age'].mean(), inplace=True)

# Fill missing embarked with mode
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# Drop deck column (too many missing values)
df.drop(columns=['deck'], inplace=True)

# ===============================
# Exploratory Data Analysis
# ===============================

# Survival Count
sns.countplot(x='survived', data=df)
plt.title("Survival Count")
plt.show()

# Survival by Gender
sns.countplot(x='sex', hue='survived', data=df)
plt.title("Survival by Gender")
plt.show()

# Survival by Passenger Class
sns.countplot(x='pclass', hue='survived', data=df)
plt.title("Survival by Passenger Class")
plt.show()

# Age Distribution
sns.histplot(df['age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Save cleaned dataset
df.to_csv("cleaned_titanic.csv", index=False)

print("EDA completed successfully")
