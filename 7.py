import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# i. Read dataset
data = pd.read_csv("Churn_Modelling.csv")

# ii. Features and target
X = data.drop(["RowNumber", "CustomerId", "Surname", "Exited"], axis=1)
y = data["Exited"]

# Encode categorical features
X = pd.get_dummies(X, columns=["Geography", "Gender"], dtype=int)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# iii. Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# iv. Build neural network
model = Sequential([
    Dense(16, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=["accuracy"])

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

# v. Evaluate model
pred = (model.predict(X_test, verbose=0) > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
