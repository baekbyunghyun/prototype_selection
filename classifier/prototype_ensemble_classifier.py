import time
import numpy as np
from random import *

from sklearn import neighbors
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from prototype_selector import PrototypeSelector


class PrototypeEnsembleClassifier:
    def __init__(self, dataset=None, type_of_prototype_selection=None,
                 num_of_classifier=1, tau=0.1, radius=0.1, k=1):
        self.dataset = dataset
        self.type_of_prototype_selection = type_of_prototype_selection
        self.num_of_classifier = num_of_classifier
        self.tau = tau
        self.radius = radius
        self.k = k

        self.X_train, self.X_test, self.y_train, self.y_test = self._split_train_test()

        self.voting_classifier = None
        self.classifier_list = list()
        self.num_of_prototypes = list()
        self.prototype_selection_times = list()
        self.learning_times = list()

    def _split_train_test(self):
        if self.dataset is None:
            raise Exception('Dataset is None.')

        X_train, X_test, y_train, y_test = train_test_split(
            self.dataset.X, self.dataset.y, test_size=0.1, random_state=42)

        return X_train, X_test, y_train, y_test

    def fit(self):
        for i in range(0, self.num_of_classifier):
            # print('{0}\' classifier is loading...'.format(i))

            X, y, elapsed = self._select_prototype_set()
            self.num_of_prototypes.append(len(X))
            self.prototype_selection_times.append(elapsed)

            classifier, elapsed = self._learn_dt(X, y)
            self.learning_times.append(elapsed)

            self.classifier_list.append(classifier)

    def _select_prototype_set(self):
        X, y = shuffle(self.X_train, self.y_train, random_state=randint(0, 100))

        selector = PrototypeSelector(algorithm=self.type_of_prototype_selection,
                                     X=X,
                                     y=y,
                                     parameter_tau=self.tau,
                                     parameter_radius=self.radius)

        start_time = time.time()

        X_prototype, y_prototype = selector.select()

        elapsed = time.time() - start_time

        return X_prototype, y_prototype, elapsed

    def _learn_dt(self, X, y):
        classifier = DecisionTreeClassifier(random_state=0)

        start_time = time.time()

        classifier.fit(X, y)

        elapsed = time.time() - start_time

        return classifier, elapsed

    def _learn_nn(self, X, y):
        classifier = neighbors.KNeighborsClassifier(self.k, weights='uniform', algorithm='brute', p=2)

        start_time = time.time()

        classifier.fit(X, y)

        elapsed = time.time() - start_time

        return classifier, elapsed

    def predict(self):
        classifier_predictions = list()

        for classifier in self.classifier_list:
            classifier_predictions.append(classifier.predict(self.X_test))

        classifier_predictions = np.asarray(classifier_predictions)
        classifier_predictions = classifier_predictions.transpose()
        classifier_predictions = classifier_predictions.tolist()

        predictions = list()
        for prediction in classifier_predictions:
            predictions.append(max(set(prediction), key=prediction.count))

        return accuracy_score(self.y_test, predictions)
