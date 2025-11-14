from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def make_logreg(random_state=42):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
    ])

def make_rf(random_state=42):
    return Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced",
            random_state=random_state
        ))
    ])

def make_gb(random_state=42):
    return Pipeline([
        ("clf", GradientBoostingClassifier(random_state=random_state))
    ])

def get_model(name: str, random_state=42):
    name = name.lower()
    if name in ["lr", "logreg", "logistic", "logistic_regression"]:
        return make_logreg(random_state)
    if name in ["rf", "random_forest"]:
        return make_rf(random_state)
    if name in ["gb", "gbrt", "gbdt", "gradient_boosting"]:
        return make_gb(random_state)
    raise ValueError(f"Unknown model name: {name}")
