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

data = fetch_openml("titanic", version=1, as_frame=True)
x = data.data

print(x.isnull().sum())

x = data.data[["age", "fare", "embarked", "sex"]].copy()
y = data.target

x["age"] = x["age"].fillna(x["age"].mean())
x["fare"] = x["fare"].fillna(x["fare"].mean())
x["embarked"] = x["embarked"].fillna(x["embarked"].mode()[0])

x["sex"] = LabelEncoder().fit_transform(x["sex"])
x["embarked"] = LabelEncoder().fit_transform(x["embarked"])

print(x.head())
print(x.isnull().sum())

model1 = KNeighborsClassifier(n_neighbors=3)
model2 = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

score1 = cross_val_score(model1, x, y, cv=cv, scoring="accuracy")
score2 = cross_val_score(model2, x, y, cv=cv, scoring="accuracy")

print("Without Scaling Scores:", score1)
print("With Scaling Scores:", score2)
print("Without Scaling:", round(score1.mean(), 3))
print("With Scaling:", round(score2.mean(), 3))