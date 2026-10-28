
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

# Load dataset
data = fetch_openml("titanic", version=1, as_frame=True)
X = data.data[["age", "fare", "embarked", "sex"]].copy()
y = data.target

# Preprocessing
X[["age", "fare"]] = SimpleImputer(strategy="mean").fit_transform(X[["age", "fare"]])
X["embarked"] = X["embarked"].fillna(X["embarked"].mode()[0])
X = pd.get_dummies(X, columns=["sex", "embarked"], dtype=int)

# Define both models
model1 = KNeighborsClassifier(n_neighbors=3)
model2 = make_pipeline(StandardScaler(),
                       KNeighborsClassifier(n_neighbors=3))

# 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

score1 = cross_val_score(model1, X, y, cv=cv, scoring="accuracy")
score2 = cross_val_score(model2, X, y, cv=cv, scoring="accuracy")

# Compare performance
print("Without Scaling:", round(score1.mean(), 3))
print("With Scaling:", round(score2.mean(), 3))
print("Without Scaling Scores:", score1)
print("With Scaling Scores:", score2)
