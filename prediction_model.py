import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


class GetModel:
    def __init__(self, dataset: pd.DataFrame = None) -> None:
        self.y = dataset["assessment"]
        self.X = dataset.drop(["assessment"], axis=1)
        self.__create_model()

    @classmethod
    def from_csv(cls, csv_path: str = None):
        return cls(pd.read_csv(csv_path))

    def __create_model(self):
        # self.model = LDA()
        self.model = LinearRegression()
        # self.model = GaussianNB()
        # self.model = SVC()
        # self.model = LogisticRegression()
        # self.model = DecisionTreeClassifier()
        self.model.fit(self.X.values, self.y)
        # cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=2, random_state=1)
        # scores = cross_val_score(
        #     self.model, self.X, self.y, scoring="accuracy", cv=cv, n_jobs=-1
        # )
        # self.model = LinearRegression()
        # self.model.fit(self.X.values, self.y)
        # self.mean_score = round(np.mean(scores), 4)
        # self.standar_deviation = round(np.std(scores), 4)

    def get_prediction(self, test_data=None):
        return self.model.predict(test_data)

    def get_model(self):
        return self.model

    # def get_scores(self):
    #     return [self.mean_score, self.standar_deviation]
