from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class ModelOutputs:
    proba_up: float
    proba_down: float
    edge: float


class IntradayEnsembleModel:
    """Two-model ensemble for robust intraday direction probabilities."""

    def __init__(self, random_state: int = 7) -> None:
        self.linear = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=800, C=0.6, random_state=random_state)),
            ]
        )
        self.tree = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=8,
                        max_depth=7,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        self.fitted = False
        self.class_prior_up = 0.5

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        classes = pd.Series(y).value_counts()
        if classes.shape[0] < 2:
            # Degenerate label periods can happen intraday; keep strategy neutral.
            self.class_prior_up = float(pd.Series(y).mean()) if len(y) else 0.5
            self.fitted = False
            return
        self.linear.fit(X, y)
        self.tree.fit(X, y)
        self.class_prior_up = float(pd.Series(y).mean())
        self.fitted = True

    def predict_one(self, row: pd.Series) -> ModelOutputs:
        if not self.fitted:
            up = self.class_prior_up
            return ModelOutputs(proba_up=up, proba_down=1 - up, edge=up - (1 - up))

        sample = row.to_frame().T
        up_linear = self.linear.predict_proba(sample)[0, 1]
        up_tree = self.tree.predict_proba(sample)[0, 1]
        # Blend smooth linear regime and nonlinear interactions.
        proba_up = float(np.clip(0.45 * up_linear + 0.55 * up_tree, 0.0, 1.0))
        proba_down = 1.0 - proba_up
        return ModelOutputs(
            proba_up=proba_up,
            proba_down=proba_down,
            edge=proba_up - proba_down,
        )
