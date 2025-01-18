import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("data/Cleaned-Data.csv")

# 1. Basic statistics
print(df.describe())

# 2. Correlation analysis for numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
correlation_matrix = df[numerical_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Numerical Variables')
plt.show()

# 3. Distribution of PCOS
plt.figure(figsize=(8, 6))
df['PCOS'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribution of PCOS')
plt.show()

# 4. Age distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=True)
plt.title('Age Distribution')
plt.show()

# 5. BMI calculation and distribution
df['BMI'] = df['Weight_kg'] / ((df['Height_ft'] * 0.3048) ** 2)
plt.figure(figsize=(10, 6))
sns.histplot(df['BMI'], kde=True)
plt.title('BMI Distribution')
plt.show()

# 6. Relationship between BMI and PCOS
plt.figure(figsize=(10, 6))
sns.boxplot(x='PCOS', y='BMI', data=df)
plt.title('BMI vs PCOS')
plt.show()

# 7. Relationship between exercise frequency and PCOS
plt.figure(figsize=(12, 6))
sns.countplot(x='Exercise_Frequency', hue='PCOS', data=df)
plt.title('Exercise Frequency vs PCOS')
plt.xticks(rotation=45)
plt.show()

# 8. Diet patterns for PCOS vs non-PCOS
diet_cols = [col for col in df.columns if col.startswith('Diet_')]
df_diet = df[diet_cols + ['PCOS']]
df_diet_melted = pd.melt(df_diet, id_vars=['PCOS'], var_name='Diet_Category', value_name='Consumption')

plt.figure(figsize=(14, 8))
sns.boxplot(x='Diet_Category', y='Consumption', hue='PCOS', data=df_diet_melted)
plt.title('Diet Patterns: PCOS vs Non-PCOS')
plt.xticks(rotation=90)
plt.show()

# 9. Stress level and PCOS
plt.figure(figsize=(10, 6))
sns.countplot(x='Stress_Level', hue='PCOS', data=df)
plt.title('Stress Level vs PCOS')
plt.show()

# 10. Menstrual irregularity and PCOS
plt.figure(figsize=(10, 6))
sns.countplot(x='Menstrual_Irregularity', hue='PCOS', data=df)
plt.title('Menstrual Irregularity vs PCOS')
plt.show()
