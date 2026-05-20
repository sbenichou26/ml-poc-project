"""Train and save all models to the models/ directory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.svm import SVC

from config import MODELS_DIR
from data import load_dataset_split

NUMERICAL_COLS = [
    "Age",
    "Technology_Usage_Hours",
    "Social_Media_Usage_Hours",
    "Gaming_Hours",
    "Screen_Time_Hours",
    "Sleep_Hours",
    "Physical_Activity_Hours",
]

ORDINAL_COLS = ["Stress_Level"]
ORDINAL_CATEGORIES = [["Low", "Medium", "High"]]

CATEGORICAL_COLS = ["Gender", "Support_Systems_Access", "Work_Environment_Impact", "Online_Support_Usage"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERICAL_COLS),
        ("ord", OrdinalEncoder(categories=ORDINAL_CATEGORIES), ORDINAL_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
    ]
)

MODELS = {
    "log_reg": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=7),
    "svm": SVC(kernel="rbf", random_state=42),
}


def train_and_save():
    print("Chargement des données...")
    X_train, X_test, y_train, y_test = load_dataset_split()

    for model_key, classifier in MODELS.items():
        print(f"Entraînement : {model_key}...")
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
        pipeline.fit(X_train, y_train)

        save_path = MODELS_DIR / f"{model_key}.joblib"
        joblib.dump(pipeline, save_path)
        print(f"  Sauvegardé : {save_path}")

    print("\nTous les modèles ont été entraînés et sauvegardés.")


if __name__ == "__main__":
    train_and_save()
