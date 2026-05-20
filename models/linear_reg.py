"""Modèle 1 — Chain Linear Regression

Chaîne causale complète :
  screen_time ──Stage 1──→ stress, productivité, sommeil, sport
                ──Stage 2──→ mental_wellness

Stage 1 : LinearRegression (screen_time → chaque intermédiaire)
Stage 2 : LinearRegression (intermédiaires → mental_wellness)

Pertinent car toutes les relations dans la chaîne sont fortement linéaires.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from chain_model import ChainRegressor
from config import MODELS_DIR
from data import load_dataset_split

model = ChainRegressor(stage2_model=LinearRegression())

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_dataset_split()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"R²   : {r2_score(y_test, y_pred):.4f}")
    print(f"MAE  : {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    joblib.dump(model, MODELS_DIR / "linear_reg.joblib")
    print("Sauvegardé : models/linear_reg.joblib")
