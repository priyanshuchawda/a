# Write a program to implement k-Nearest Neighbors (K-NN) on any dataset from repository and perform following operations.
# i. Preprocess the dataset (handle missing values, encode categoricalfeatures).
# ii. Apply feature scaling.
# iii. Train both models and evaluate using cross-validation (k-fold).
# iv. Compare performance metrics.

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

data = load_iris()
x = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("Dataset Shape:", x.shape)
print(x.isnull().sum())

# 3. Create models
model1 = KNeighborsClassifier(n_neighbors=3)

model2 = make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=3))

# 4. Apply 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

score1 = cross_val_score(model1, x, y, cv=cv, scoring="accuracy")
score2 = cross_val_score(model2, x, y, cv=cv, scoring="accuracy")

print("Without Scaling Scores:", score1)
print("With Scaling Scores:", score2)

print("Without Scaling:", round(score1.mean(), 3))
print("With Scaling:", round(score2.mean(), 3))
