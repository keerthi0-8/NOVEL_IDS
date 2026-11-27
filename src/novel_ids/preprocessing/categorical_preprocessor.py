
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class CategoricalPreprocessor:
    """Dataset-aware categorical encoding with training-only vocabularies."""

    vocabularies: dict = field(default_factory=dict)
    feature_names: list | None = None
    fitted: bool = False

    UNKNOWN_TOKEN = "UNKNOWN"
    UNKNOWN_ID = 0

    def fit(self, X_train: pd.DataFrame):
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")

        if X_train.empty:
            raise ValueError("Training data cannot be empty.")

        self.feature_names = list(X_train.columns)
        self.vocabularies = {}

        for feature in self.feature_names:
            values = X_train[feature].dropna().astype(str)
            categories = sorted(
                value
                for value in values.unique()
                if value != self.UNKNOWN_TOKEN
            )

            # ID 0 is reserved for missing or unseen categories.
            self.vocabularies[feature] = {
                category: index + 1
                for index, category in enumerate(categories)
            }

        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame):
        if not self.fitted:
            raise RuntimeError("Call fit() before transform().")

        if list(X.columns) != self.feature_names:
            raise ValueError(
                "Input categorical columns or order do not match training data."
            )

        encoded = pd.DataFrame(index=X.index)
        M_miss = pd.DataFrame(index=X.index)

        for feature in self.feature_names:
            values = X[feature].astype("string")

            M_miss[feature] = values.isna().astype("float32")

            values = values.fillna(self.UNKNOWN_TOKEN).astype(str)

            mapping = self.vocabularies[feature]

            encoded[feature] = (
                values.map(mapping)
                .fillna(self.UNKNOWN_ID)
                .astype("int64")
            )

        return encoded, M_miss

    def fit_transform(self, X_train: pd.DataFrame):
        self.fit(X_train)
        return self.transform(X_train)
