from data import load_dataset


def test_load_dataset_shape():
    X, y, names = load_dataset()

    assert X.shape[0] == len(y)
    assert X.shape[1] == 30
    assert names == ["malignant", "benign"]
