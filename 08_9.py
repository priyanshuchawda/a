#
# Write a program to implement k-Nearest Neighbors (K-NN) on any dataset
# from repository and perform following operations.
# i. Preprocess the dataset (handle missing values, encode categorical
# features).
# ii. Apply feature scaling.
# iii. Train both models and evaluate using cross-validation (k-fold).
# iv. Compare performance metrics.


import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Load dataset
data = fetch_openml("titanic", version=1, as_frame=True)
X = data.data[["age", "fare", "embarked", "sex"]].copy()
y = data.target

# Check missing values
print("Missing values before preprocessing:")
print(X.isnull().sum())

# Handle missing values
X["age"] = X["age"].fillna(X["age"].mean())
X["fare"] = X["fare"].fillna(X["fare"].mean())
X["embarked"] = X["embarked"].fillna(X["embarked"].mode()[0])

# Encode categorical features
X["sex"] = LabelEncoder().fit_transform(X["sex"])
X["embarked"] = LabelEncoder().fit_transform(X["embarked"])

print("\nPreprocessed Data:")
print(X.head())
print("\nMissing values after preprocessing:")
print(X.isnull().sum())

# Define models
model1 = KNeighborsClassifier(n_neighbors=3)
model2 = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=3)
)

# 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

score1 = cross_val_score(model1, X, y, cv=cv, scoring="accuracy")
score2 = cross_val_score(model2, X, y, cv=cv, scoring="accuracy")

# Compare performance
print("\nWithout Scaling:", round(score1.mean(), 3))
print("With Scaling:", round(score2.mean(), 3))
print("Without Scaling Scores:", score1)
print("With Scaling Scores:", score2)
