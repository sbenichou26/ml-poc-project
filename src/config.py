from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PLOTS_DIR = PROJECT_ROOT / "plots"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"

for dir in [
    DATA_DIR,
    LOGS_DIR,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    PLOTS_DIR,
    RESULTS_DIR,
    SCRIPTS_DIR,
    TESTS_DIR,
]:
    dir.mkdir(exist_ok=True)

ENV_FILE = PROJECT_ROOT / ".env"
APP_ENTRYPOINT = PROJECT_ROOT / "src" / "app.py"
MODEL_METRICS_FILE = RESULTS_DIR / "model_metrics.csv"

STREAMLIT_HOST = "localhost"
STREAMLIT_PORT = 8501

MODELS = {
    "linear_reg": {
        "name": "Ridge Regression",
        "description": "Modèle baseline linéaire régularisé.",
        "path": MODELS_DIR / "linear_reg.joblib",
    },
    "random_forest": {
        "name": "Random Forest",
        "description": "Ensemble de 300 arbres de décision.",
        "path": MODELS_DIR / "random_forest.joblib",
    },
    "gradient_boosting": {
        "name": "Gradient Boosting",
        "description": "Arbres construits en séquence, chacun corrige le précédent.",
        "path": MODELS_DIR / "gradient_boosting.joblib",
    },
    "knn": {
        "name": "XGBoost",
        "description": "Gradient boosting optimisé, très performant sur données tabulaires.",
        "path": MODELS_DIR / "knn.joblib",
    },
    "svr": {
        "name": "LightGBM",
        "description": "Gradient boosting rapide basé sur des histogrammes.",
        "path": MODELS_DIR / "svr.joblib",
    },
}
