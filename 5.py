# 5.
# Write a program to implement k-Nearest Neighbors (K-NN) on any dataset
# from repository and perform following operations.
# i. Preprocess the dataset (handle missing values, encode categorical
# features).
# ii. Apply feature scaling.
# iii. Train both models and evaluate using cross-validation (k-fold).
# iv. Compare performance metrics.

from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Handle missing values
X = SimpleImputer(strategy="mean").fit_transform(X)

# Model 1: K = 3
m1 = make_pipeline(StandardScaler(),
                   KNeighborsClassifier(n_neighbors=3))

s1 = cross_val_score(m1, X, y, cv=5)

# Model 2: K = 5
m2 = make_pipeline(StandardScaler(),
                   KNeighborsClassifier(n_neighbors=5))

s2 = cross_val_score(m2, X, y, cv=5)

# Compare accuracy
print("K = 3, Accuracy:", s1.mean())
print("K = 5, Accuracy:", s2.mean())
