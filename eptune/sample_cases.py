# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/05_sample_cases.ipynb (unless otherwise specified).

__all__ = ['Digits', 'DigitsCV']

# Cell
from .sklearn import ScikitLearner, ScikitLearnerCV

import sklearn
from sklearn import datasets
from sklearn import model_selection
from sklearn import svm
from sklearn import ensemble
from sklearn import linear_model


class Digits(ScikitLearner):
    "Sample case to use the helper class `ScikitLearner`."

    def __init__(self, learner, seed=42):
        data = sklearn.datasets.load_digits()
        super().__init__(
            learner,
            *model_selection.train_test_split(
                data["data"], data["target"], test_size=0.2,
                random_state=seed))


# Cell
class DigitsCV(ScikitLearnerCV):
    "Sample case to use the helper class `ScikitLearner`."

    def __init__(self, learner):
        data = sklearn.datasets.load_digits()
        super().__init__(learner, X=data["data"], y=data["target"])