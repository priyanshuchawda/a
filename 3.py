
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif, SelectPercentile
from sklearn.decomposition import PCA

# Load dataset
data = load_iris()
X = data.data
y = data.target

# 1. SelectKBest
kbest = SelectKBest(score_func=f_classif, k=2)
X1 = kbest.fit_transform(X, y)
print("SelectKBest:", X1.shape)

# 2. SelectPercentile
percentile = SelectPercentile(score_func=f_classif, percentile=50)
X2 = percentile.fit_transform(X, y)
print("SelectPercentile:", X2.shape)

# 3. PCA
pca = PCA(n_components=2)
X3 = pca.fit_transform(X)
print("PCA:", X3.shape)