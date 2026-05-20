"""Student-owned dataset loading contract."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_DIR

DATASET_PATH     = DATA_DIR / "ScreenTime vs MentalWellness.csv"
TARGET_COL       = "mental_wellness_index_0_100"
CATEGORICAL_COLS = ["gender", "occupation", "work_mode"]
SCREEN_THRESHOLD = 12.0


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame()

    # Features brutes
    X["screen_time_hours"]         = df["screen_time_hours"]
    X["leisure_screen_hours"]      = df["leisure_screen_hours"]
    X["work_screen_hours"]         = df["work_screen_hours"]
    X["sleep_hours"]               = df["sleep_hours"]
    X["sleep_quality_1_5"]         = df["sleep_quality_1_5"]
    X["stress_level_0_10"]         = df["stress_level_0_10"]
    X["productivity_0_100"]        = df["productivity_0_100"]
    X["exercise_minutes_per_week"] = df["exercise_minutes_per_week"]
    X["age"]                       = df["age"]
    X["gender"]                    = df["gender"]
    X["occupation"]                = df["occupation"]
    X["work_mode"]                 = df["work_mode"]

    # Feature engineering
    X["work_ratio"]            = df["work_screen_hours"] / df["screen_time_hours"]
    X["leisure_ratio"]         = df["leisure_screen_hours"] / df["screen_time_hours"]
    X["sleep_score"]           = df["sleep_hours"] * df["sleep_quality_1_5"]
    X["screen_per_sleep"]      = df["screen_time_hours"] / df["sleep_hours"]
    X["active_balance"]        = df["exercise_minutes_per_week"] - (df["screen_time_hours"] * 10)
    X["screen_over_threshold"] = np.maximum(0, df["screen_time_hours"] - SCREEN_THRESHOLD)

    return X


def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    """Return (X_train, X_test, y_train, y_test) for model evaluation."""
    df = pd.read_csv(DATASET_PATH)
    X  = _build_features(df)
    y  = df[TARGET_COL].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
