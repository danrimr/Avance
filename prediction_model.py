"""Docs"""

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# from sklearn.linear_model import LinearRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression

# from sklearn.tree import DecisionTreeClassifier


class Model:
    """doc"""

    def __init__(self, dataset: pd.DataFrame = None) -> None:
        self.y = dataset["assessment"]
        self.X = dataset.drop(["assessment"], axis=1)
        self.__create_model()

    @classmethod
    def from_csv(cls, csv_path: str = None):
        """Docs"""
        return cls(pd.read_csv(csv_path))

    def lda_reduction(self):
        ...

    def __create_model(self):
        self.model = LDA()
        self.model.fit(self.X.values, self.y)

    def get_scores(self):
        ...

    def get_prediction(self, test_data=None):
        """Docs"""
        return self.model.predict(test_data)

    def get_model(self):
        """Docs"""
        return self.model
