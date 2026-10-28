#
# Write a program to implement k-Nearest Neighbors (K-NN) on any dataset
# from repository and perform following operations.
# i. Preprocess the dataset (handle missing values, encode categorical
# features).


import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load Titanic dataset
data = fetch_openml("titanic", version=1, as_frame=True)
X = data.data[["age", "fare", "embarked", "sex"]].copy()
y = data.target

# 1. Handle missing values
X["age"] = X["age"].fillna(X["age"].mean())
X["fare"] = X["fare"].fillna(X["fare"].mean())
X["embarked"] = X["embarked"].fillna(X["embarked"].mode()[0])

# 2. Encode categorical features
X["sex"] = LabelEncoder().fit_transform(X["sex"])
X["embarked"] = LabelEncoder().fit_transform(X["embarked"])

print("Preprocessed Data:")
print(X.head())
print("\nMissing Values:\n", X.isnull().sum())

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Apply K-NN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 6. Predict and evaluate
pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, pred))
