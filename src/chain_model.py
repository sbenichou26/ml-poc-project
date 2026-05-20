"""Chain Regressor — modèle en deux étapes.

Stage 1 : screen_time → [stress, productivity, sleep_quality, sleep_hours, exercise]
Stage 2 : [stress_pred, productivity_pred, ...] → mental_wellness_index
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from data import DATASET_PATH, FEATURE_COLS, INTERMEDIATE_COLS, TARGET_COL


class ChainRegressor(BaseEstimator, RegressorMixin):
    """Régression en chaîne : screen_time → intermédiaires → santé mentale."""

    def __init__(self, stage2_model):
        self.stage2_model = stage2_model
        self.stage1_models_ = {}
        self.scaler_ = StandardScaler()

    def fit(self, X, y):
        df = pd.read_csv(DATASET_PATH)

        X_screen = df[FEATURE_COLS].values
        X_screen_scaled = self.scaler_.fit_transform(X_screen)

        # Stage 1 : apprendre screen_time → chaque variable intermédiaire
        for col in INTERMEDIATE_COLS:
            m = LinearRegression()
            m.fit(X_screen_scaled, df[col].values)
            self.stage1_models_[col] = m

        # Générer les prédictions intermédiaires sur tout le dataset
        X_intermediates = self._predict_intermediates(X_screen_scaled)

        # Stage 2 : apprendre intermédiaires → mental_wellness
        self.stage2_model.fit(X_intermediates, df[TARGET_COL].values)

        return self

    def predict(self, X):
        X_scaled = self.scaler_.transform(X.values if hasattr(X, "values") else X)
        X_intermediates = self._predict_intermediates(X_scaled)
        return self.stage2_model.predict(X_intermediates)

    def _predict_intermediates(self, X_scaled):
        return np.column_stack([
            self.stage1_models_[col].predict(X_scaled)
            for col in INTERMEDIATE_COLS
        ])
