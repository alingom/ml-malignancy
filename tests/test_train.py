import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from data import load_dataset
 
def test_load_dataset_shape():
    X, y, names = load_dataset()
    assert X.shape[0] == len(y)
    assert len(names) == 2