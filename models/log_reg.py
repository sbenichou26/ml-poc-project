"""Modèle 1 — Logistic Regression (baseline)

C'est quoi ?
    Le modèle le plus simple. Il cherche une frontière linéaire entre les classes.
    Sert de référence : si les autres ne font pas mieux, il y a un problème.

Hyperparamètres :
    - max_iter=1000 : nombre max d'itérations pour converger
    - random_state=42 : pour reproduire les mêmes résultats
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from config import MODELS_DIR
from data import load_dataset_split

NUMERICAL_COLS = [
    "Age", "Technology_Usage_Hours", "Social_Media_Usage_Hours",
    "Gaming_Hours", "Screen_Time_Hours", "Sleep_Hours", "Physical_Activity_Hours",
]
ORDINAL_COLS = ["Stress_Level"]
CATEGORICAL_COLS = ["Gender", "Support_Systems_Access", "Work_Environment_Impact", "Online_Support_Usage"]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), NUMERICAL_COLS),
    ("ord", OrdinalEncoder(categories=[["Low", "Medium", "High"]]), ORDINAL_COLS),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
])

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_dataset_split()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=["Excellent", "Fair", "Good", "Poor"]))
    joblib.dump(model, MODELS_DIR / "log_reg.joblib")
    print("Sauvegardé : models/log_reg.joblib")
