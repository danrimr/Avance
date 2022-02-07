"""Docs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

# from sklearn.tree import DecisionTreeClassifier as DTC

from sklearn.naive_bayes import GaussianNB


class Classifier:
    """Docs."""

    def __init__(self, dataset: pd.DataFrame) -> None:
        self.dataset = dataset
        self.y_data: pd.DataFrame
        self.x_data: pd.DataFrame
        self.lda: LDA
        self.model: GaussianNB

    def make_classifier(self) -> None:
        """Docs."""

        self.__dataset_preprocessiong()
        self.__lda_reduction()
        self.__training_model()

    def __dataset_preprocessiong(self) -> None:
        self.y_data = self.dataset["class"]
        self.x_data = self.dataset.drop(["class"], axis=1)

    def __lda_reduction(self) -> None:
        self.lda = LDA(n_components=2)
        self.x_data = self.lda.fit_transform(self.x_data.values, self.y_data)

    def __training_model(self) -> None:
        self.model = GaussianNB()
        self.model.fit(self.x_data, self.y_data)

    def get_model_score(self) -> list:
        """Docs."""

        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        scores = cross_val_score(
            self.model,
            self.x_data,
            self.y_data,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            error_score="raise",
        )

        mean_score = np.mean(scores)
        std_score = np.std(scores)

        return list((round(mean_score, 4), round(std_score, 4)))

    def get_prediction(self, test_data: np.ndarray) -> np.ndarray:
        """Docs."""

        data = self.lda.transform(test_data)
        return self.model.predict(data)

    def get_model(self):
        """Docs."""
        return self.model

    @classmethod
    def from_csv(cls, csv_path: str) -> Classifier:
        """Docs."""
        return cls(pd.read_csv(csv_path))
