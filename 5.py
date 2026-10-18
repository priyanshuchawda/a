
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Handle missing values
X = SimpleImputer(strategy="mean").fit_transform(X)

# Encode categorical target
y = LabelEncoder().fit_transform(y)

# 5-fold cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Train and compare K-NN models
for k in [3, 5]:
    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=k)
    )

    scores = cross_val_score(model, X, y, cv=cv)

    print("K =", k)
    print("Fold scores:", scores)
    print("Mean accuracy:", scores.mean())
