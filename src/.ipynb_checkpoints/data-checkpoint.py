import pandas as pd
from sklearn.datasets import load_breast_cancer
 
def load_dataset():
    ds = load_breast_cancer()
    X = pd.DataFrame(ds.data, columns=ds.feature_names)
    y = pd.Series(ds.target, name="target")
    target_names = list(ds.target_names)
    return X, y, target_names

