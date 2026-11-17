import numpy as np
import sklearn
from sklearn.cluster import HDBSCAN
print("sklearn", sklearn.__version__)
rng = np.random.default_rng(0)
X = np.vstack([rng.normal(size=(300, 5)) + c for c in ([0]*5, [8]*5, [16]*5)])
m = HDBSCAN(min_cluster_size=20, min_samples=5)
out = m.fit_predict(X)
print("fit_predict ->", type(out), np.unique(out))
print("attrs:", [a for a in dir(m) if a.endswith("_") and not a.startswith("__")])
print("labels_?", hasattr(m, "labels_"), "outliers_scores_?", hasattr(m, "outliers_scores_"))
