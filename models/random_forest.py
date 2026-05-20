"""Modèle 2 — Chain Random Forest

Stage 1 : LinearRegression (screen_time → intermédiaires)
Stage 2 : RandomForestRegressor (intermédiaires → mental_wellness)

Capture les interactions non-linéaires entre stress, sommeil et productivité.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from chain_model import ChainRegressor
from config import MODELS_DIR
from data import load_dataset_split

model = ChainRegressor(stage2_model=RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42))

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_dataset_split()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"R²   : {r2_score(y_test, y_pred):.4f}")
    print(f"MAE  : {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    joblib.dump(model, MODELS_DIR / "random_forest.joblib")
    print("Sauvegardé : models/random_forest.joblib")
