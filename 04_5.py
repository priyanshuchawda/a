# Write a program to perform the following operations on the selected dataset
# and display the result.
# i. Reduce dimensions using SelectKBest method.
# ii. Reduce dimensions using SelectPercentile method.
# iii. apply pca

from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif
from sklearn.decomposition import PCA
import pandas as pd

# Load dataset
data = load_iris()
x = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# 1. SelectKBest
kbest = SelectKBest(f_classif, k=2)
kbest.fit(x, y)
KBest = x[x.columns[kbest.get_support()]]

print("SelectKBest:")
print(KBest.head())
print("Columns:", list(KBest.columns))

# 2. SelectPercentile
percentile = SelectPercentile(f_classif, percentile=50)
percentile.fit(x, y)
Percentile = x[x.columns[percentile.get_support()]]

print("\nSelectPercentile:")
print(Percentile.head())
print("Columns:", list(Percentile.columns))

# Apply PCA
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

# Create new table
PCA_Data = pd.DataFrame(x_pca, columns=['PC1', 'PC2'])

print("\nPCA Data:")
print(PCA_Data.head())
print("New Shape:", PCA_Data.shape)
