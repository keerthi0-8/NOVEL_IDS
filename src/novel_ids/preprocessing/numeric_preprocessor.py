
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


@dataclass
class NumericPreprocessor:
    """
    Training-only numeric preprocessing.

    Fit only on the source training partition.
    Validation and test data must use transform().
    """

    log_features: list[str] | None = None

    def __post_init__(self):
        self.log_features = self.log_features or []
        self.medians = None
        self.scaler = None
        self.feature_names = None
        self.missing_columns = None
        self.fitted = False

    def fit(self, X_train: pd.DataFrame):
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")

        if X_train.empty:
            raise ValueError("Training data cannot be empty.")

        X = X_train.copy()

        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            raise ValueError(
                f"Categorical columns passed to numeric preprocessor: "
                f"{non_numeric}"
            )

        self.feature_names = list(X.columns)

        unknown_log_features = set(self.log_features) - set(self.feature_names)
        if unknown_log_features:
            raise ValueError(
                f"Unknown log features: {sorted(unknown_log_features)}"
            )

        X = X.replace([np.inf, -np.inf], np.nan)

        self.missing_columns = X.columns[X.isna().any()].tolist()

        for feature in self.log_features:
            observed = X[feature].dropna()

            if (observed < 0).any():
                raise ValueError(
                    f"Cannot apply log1p to negative values in {feature}."
                )

        self.medians = X.median()

        all_missing = self.medians[self.medians.isna()].index.tolist()
        if all_missing:
            raise ValueError(
                "Features entirely missing in training data: "
                f"{all_missing}"
            )

        X = X.fillna(self.medians)

        for feature in self.log_features:
            X[feature] = np.log1p(X[feature])

        self.scaler = RobustScaler()
        self.scaler.fit(X)

        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame):
        if not self.fitted:
            raise RuntimeError("Call fit() before transform().")

        if list(X.columns) != self.feature_names:
            raise ValueError(
                "Input feature columns or order do not match training data."
            )

        X = X.copy()

        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
            raise TypeError("Categorical columns cannot be transformed here.")

        X = X.replace([np.inf, -np.inf], np.nan)

        # Record missingness before imputation.
        M_miss = X.isna().astype(np.float32)

        X = X.fillna(self.medians)

        for feature in self.log_features:
            X[feature] = np.log1p(X[feature])

        values = self.scaler.transform(X)

        X_scaled = pd.DataFrame(
            values,
            columns=self.feature_names,
            index=X.index,
        )

        if not np.isfinite(X_scaled.to_numpy()).all():
            raise ValueError("Preprocessing produced NaN or Inf values.")

        return X_scaled, M_miss

    def fit_transform(self, X_train: pd.DataFrame):
        self.fit(X_train)
        return self.transform(X_train)
