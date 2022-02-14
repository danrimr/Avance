from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

# from sklearn.tree import DecisionTreeClassifier as DTC

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression


class Classifier:
    """Permits to create, train and test a model ["LinearRegression", "GaussianNB"]."""

    def __init__(self, dataset: pd.DataFrame, model: str = "linear") -> None:
        self.dataset = dataset
        self.y_data: pd.DataFrame
        self.x_data: pd.DataFrame
        self.lda = LDA(n_components=2)
        self.model = LinearRegression() if model == "linear" else GaussianNB()

    def make_classifier(self) -> None:
        """Creates and trains the model."""

        self.__dataset_preprocessiong()
        self.__lda_reduction()
        self.__training_model()

    def __dataset_preprocessiong(self) -> None:
        self.y_data = self.dataset["class"]
        self.x_data = self.dataset.drop(["class"], axis=1)

    def __lda_reduction(self) -> None:
        self.x_data = self.lda.fit_transform(self.x_data.values, self.y_data)

    def __training_model(self) -> None:
        self.model.fit(self.x_data, self.y_data)

    def get_model_score(self) -> list:
        """Compute the mean score and standard deviation from the GaussianNB model."""

        if not isinstance(self.model, LinearRegression):

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
        else:
            mean_score = None
            std_score = None

        return list((mean_score, std_score))

    def get_prediction(self, test_data: np.ndarray) -> list:
        """Predicts the class of the input data."""

        data = self.lda.transform(test_data)
        prediction = self.model.predict(data)
        return list((prediction))

    def get_model(self) -> Classifier.model:
        """Returns the current model."""
        return self.model

    @classmethod
    def from_csv(cls, csv_path: str, model: str = "linear") -> Classifier:
        """Creates the model from a cvs file."""
        return cls(pd.read_csv(csv_path), model)
