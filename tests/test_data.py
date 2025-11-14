import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from train import train_and_save
 
def test_train_and_save(tmp_path):
    out = tmp_path / "model.joblib"
    meta = train_and_save("lr", str(out), seed=123)
    assert out.exists()
    assert 0.0 <= meta["test_roc_auc"] <= 1.0
 