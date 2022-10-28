import numpy as np


class HyperRectangle:
    def __init__(self, index=None, x=None, y=None):
        self.index = index
        self.x_min = x
        self.x_max = x
        self.y = y

        self.covered_data_indexes = set([index])
        self.x_mid = (self.x_max + self.x_min) / 2.0
        self.R = self.x_max - self.x_min

    def set_covered_data(self, index=None, x=None):
        self.x_min = np.minimum(self.x_min, x)
        self.x_max = np.maximum(self.x_max, x)
        self.R = (self.x_max - self.x_min) / 2.0
        self.x_mid = (self.x_max + self.x_min) / 2.0
        self.covered_data_indexes.add(index)

    def __str__(self):
        np.set_printoptions(suppress=True)

        msg = '----------- Overview on a rectangle -----------\n'
        msg += 'index: {0}, min: {1}, max: {2}\n'.format(self.index, self.x_min, self.x_max)
        msg += 'middle: {0}, R: {1}\n'.format(self.x_mid, self.R)
        msg += 'covered data set: {0}, y: {1}\n'.format(self.covered_data_indexes, self.y)
        msg += '------------------------------------------------'

        return msg


class DataPerClass:
    def __init__(self):
        self.X = list()
        self.target = None
        self.inner_outline = None
        self.outer_outline = None


class Outline:
    def __init__(self):
        self.x_min = None
        self.x_max = None


class HyperRectangleSelectorV40:
    def __init__(self, X=None, y=None, tau=1.0, delta=1.0):
        self.X = X
        self.y = y
        self.tau = tau
        self.delta = delta
        self.temp_index = 0

        self.data_per_class_list = list()
        self.hyper_rectangles = list()

        self.prototypes = None
        self.targets = None

    def greedy(self):
        data_per_class_list = self._split_data_by_class()
        for data_per_class in data_per_class_list:
            outline = self._get_inner_outline(data_per_class.X)
            data_per_class.inner_outline = outline

            outline = self._get_outer_outline(data_per_class.X)
            data_per_class.outer_outline = outline

            self.hyper_rectangles = self.hyper_rectangles + self._generate_hyper_rectangle(data_per_class)

        self.prototypes = list()
        self.targets = list()

        for hr in self.hyper_rectangles:
            y_prime = hr.y
            x_prime = np.mean(self.X[list(hr.covered_data_indexes)], axis=0)

            self.prototypes.append(x_prime)
            self.targets.append(y_prime)

        self.prototypes = np.array(self.prototypes, ndmin=2)
        self.targets = np.array(self.targets, ndmin=1)

    def _split_data_by_class(self):
        data_per_class_list = list()

        classes = np.unique(self.y)
        for y in classes:
            data_per_class = DataPerClass()
            data_per_class.target = y

            data_per_class_list.append(data_per_class)

        for x, y in zip(self.X, self.y):
            for data_per_class in data_per_class_list:
                if data_per_class.target == y:
                    data_per_class.X.append(x)

                    break

        return data_per_class_list

    def _get_inner_outline(self, data):
        inner_outline = Outline()
        inner_outline.x_min = np.asarray(data).min(axis=0)
        inner_outline.x_max = np.asarray(data).max(axis=0)

        return inner_outline

    def _get_outer_outline(self, data):
        outer_outline = Outline()
        outer_outline.x_min = np.asarray(data).min(axis=0) - self.delta
        outer_outline.x_max = np.asarray(data).max(axis=0) + self.delta

        return outer_outline

    def _generate_hyper_rectangle(self, data_per_class):
        hyper_rectangles = list()

        for x in data_per_class.X:
            is_covered = False

            for hr in hyper_rectangles:
                if self._is_satisfy_theta(hr, x):
                    hr.set_covered_data(self.temp_index, x)
                    is_covered = True

                    break

            if is_covered is False:
                new_hr = HyperRectangle(index=self.temp_index, x=x, y=data_per_class.target)
                hyper_rectangles.append(new_hr)

            self.temp_index += 1

        return hyper_rectangles

    def _is_satisfy_theta(self, hyper_rectangle, x):
        distance = np.maximum(hyper_rectangle.x_max, x) - np.minimum(hyper_rectangle.x_min, x)

        return np.all(distance <= self.tau)

    def get_X(self):
        return self.prototypes

    def get_y(self):
        return self.targets

    def get_hyper_rectangles(self):
        return self.hyper_rectangles

    def get_number_of_prototypes(self):
        return len(self.targets)
