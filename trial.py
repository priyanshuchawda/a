
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

# Train and compare two KNN models
for k in [3, 5]:
    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=k)
    )

    scores = cross_val_score(model, X, y, cv=5)

    print("K =", k)
    print("Accuracy of each fold:", scores)
    print("Mean accuracy:", scores.mean())
    print()
