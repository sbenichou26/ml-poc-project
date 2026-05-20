"""Modèle 3 — Chain Gradient Boosting

Stage 1 : LinearRegression (screen_time → intermédiaires)
Stage 2 : GradientBoostingRegressor (intermédiaires → mental_wellness)

Arbres séquentiels qui corrigent les erreurs du précédent.
Souvent le plus performant sur données tabulaires.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from chain_model import ChainRegressor
from config import MODELS_DIR
from data import load_dataset_split

model = ChainRegressor(stage2_model=GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42))

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_dataset_split()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"R²   : {r2_score(y_test, y_pred):.4f}")
    print(f"MAE  : {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    joblib.dump(model, MODELS_DIR / "gradient_boosting.joblib")
    print("Sauvegardé : models/gradient_boosting.joblib")
