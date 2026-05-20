"""Modèle 5 — Chain SVR

Stage 1 : LinearRegression (screen_time → intermédiaires)
Stage 2 : SVR kernel RBF (intermédiaires → mental_wellness)

Très efficace sur petits datasets (400 lignes).
Le kernel RBF capture les relations non-linéaires entre variables.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import joblib
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from chain_model import ChainRegressor
from config import MODELS_DIR
from data import load_dataset_split

model = ChainRegressor(stage2_model=SVR(kernel="rbf", C=100, epsilon=0.1))

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_dataset_split()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"R²   : {r2_score(y_test, y_pred):.4f}")
    print(f"MAE  : {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    joblib.dump(model, MODELS_DIR / "svr.joblib")
    print("Sauvegardé : models/svr.joblib")
