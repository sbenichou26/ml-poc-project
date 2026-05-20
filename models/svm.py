"""Modèle 5 — Support Vector Machine (SVM)

C'est quoi ?
    Trouve la frontière qui maximise la distance entre les classes.
    Le kernel RBF lui permet de tracer des frontières courbes (non-linéaires)
    pour mieux séparer des données complexes en 4 classes.

Hyperparamètres :
    - kernel='rbf'   : type de frontière (RBF = courbe, adapté à ce problème)
    - random_state=42: reproductibilité
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
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
    ("classifier", SVC(kernel="rbf", random_state=42)),
])

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_dataset_split()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=["Excellent", "Fair", "Good", "Poor"]))
    joblib.dump(model, MODELS_DIR / "svm.joblib")
    print("Sauvegardé : models/svm.joblib")
