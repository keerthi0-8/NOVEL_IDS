
from dataclasses import dataclass, field

import pandas as pd

from .numeric_preprocessor import NumericPreprocessor
from .categorical_preprocessor import CategoricalPreprocessor


@dataclass
class DatasetPreprocessor:
    """Combines numeric and categorical preprocessing for one dataset."""

    log_features: list[str] = field(default_factory=list)

    numeric_columns: list[str] = field(default_factory=list, init=False)
    categorical_columns: list[str] = field(default_factory=list, init=False)

    numeric_processor: NumericPreprocessor | None = field(
        default=None, init=False
    )
    categorical_processor: CategoricalPreprocessor | None = field(
        default=None, init=False
    )

    feature_names: list[str] = field(default_factory=list, init=False)
    fitted: bool = field(default=False, init=False)

    def fit(self, X_train: pd.DataFrame):
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("Training input must be a pandas DataFrame.")

        if X_train.empty:
            raise ValueError("Training data cannot be empty.")

        if not X_train.columns.is_unique:
            raise ValueError("Feature names must be unique.")

        self.feature_names = list(X_train.columns)

        self.numeric_columns = X_train.select_dtypes(
            include=["number"]
        ).columns.tolist()

        self.categorical_columns = X_train.select_dtypes(
            exclude=["number"]
        ).columns.tolist()

        if self.numeric_columns:
            unknown_logs = (
                set(self.log_features) - set(self.numeric_columns)
            )

            if unknown_logs:
                raise ValueError(
                    "Log features must be numeric: "
                    f"{sorted(unknown_logs)}"
                )

            self.numeric_processor = NumericPreprocessor(
                log_features=self.log_features
            )

            self.numeric_processor.fit(
                X_train[self.numeric_columns]
            )

        if self.categorical_columns:
            self.categorical_processor = CategoricalPreprocessor()

            self.categorical_processor.fit(
                X_train[self.categorical_columns]
            )

        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame):
        if not self.fitted:
            raise RuntimeError("Call fit() before transform().")

        if list(X.columns) != self.feature_names:
            raise ValueError(
                "Input features or their order do not match training data."
            )

        result = {
            "numeric_values": pd.DataFrame(index=X.index),
            "categorical_values": pd.DataFrame(index=X.index),
            "numeric_missing_mask": pd.DataFrame(index=X.index),
            "categorical_missing_mask": pd.DataFrame(index=X.index),
            "feature_names": list(self.feature_names),
            "numeric_columns": list(self.numeric_columns),
            "categorical_columns": list(self.categorical_columns),
        }

        if self.numeric_columns:
            values, missing = self.numeric_processor.transform(
                X[self.numeric_columns]
            )
            result["numeric_values"] = values
            result["numeric_missing_mask"] = missing

        if self.categorical_columns:
            values, missing = self.categorical_processor.transform(
                X[self.categorical_columns]
            )
            result["categorical_values"] = values
            result["categorical_missing_mask"] = missing

        return result

    def fit_transform(self, X_train: pd.DataFrame):
        self.fit(X_train)
        return self.transform(X_train)
