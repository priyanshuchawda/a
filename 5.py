
# Write a program to perform the following operations on the selected dataset
# and display the result.
# iii. Reduce dimensions using PCA techniques.


from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd

# Load dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)

print("Original Data:")
print(X.head())
print("Original Shape:", X.shape)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Create new table
PCA_Data = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

print("\nPCA Data:")
print(PCA_Data.head())
print("New Shape:", PCA_Data.shape)
