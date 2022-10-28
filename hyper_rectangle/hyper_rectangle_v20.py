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


class HyperRectangleSelectorV20:
    def __init__(self, X=None, y=None, tau=1, debug=False):
        self.X = X
        self.y = y
        self.tau = tau
        self.debug = debug

        self.prototypes = None
        self.targets = None
        self.hyper_rectangles = list()

    def is_include(self, hyper_rectangle, x, y):
        distance = np.maximum(hyper_rectangle.x_max, x) - np.minimum(hyper_rectangle.x_min, x)
        is_in = np.all(distance <= self.tau)
        is_same_class = hyper_rectangle.y == y

        if is_in and is_same_class and not self.is_cover_other_class(hyper_rectangle, x):
            return True

        else:
            return False

    def is_cover_other_class(self, hyper_rectangle, target_x):
        target_x_min = np.minimum(hyper_rectangle.x_min, target_x)
        target_x_max = np.maximum(hyper_rectangle.x_max, target_x)
        target_mid = (target_x_max + target_x_min) / 2.0
        target_R = target_x_max - target_x_min\

        for x, y in zip(self.X, self.y):
            if hyper_rectangle.y == y:
                continue

            is_in = True
            for new, mid, r in zip(x, target_mid, (target_R / 2.0)):
                if abs(new - mid) > r:
                    is_in = False
                    break

            if is_in:
                return True

        return False

    def greedy(self):
        index = 0

        for x, y in zip(self.X, self.y):
            is_covered = False

            if self.debug is True:
                if index % 1000 == 0:
                    print('[DEBUG] The index for prototype selection -> {0} / {1}'.format(index, len(self.y)))

            for hr in self.hyper_rectangles:
                if self.is_include(hr, x, y):
                    hr.set_covered_data(index, x)
                    is_covered = True

                    break

            if is_covered is False:
                new_hr = HyperRectangle(index=index, x=x, y=y)
                self.hyper_rectangles.append(new_hr)

            index += 1

        self.prototypes = list()
        self.targets = list()

        for hr in self.hyper_rectangles:
            y_prime = hr.y
            x_prime = np.mean(self.X[list(hr.covered_data_indexes)], axis=0)

            self.prototypes.append(x_prime)
            self.targets.append(y_prime)

        self.prototypes = np.array(self.prototypes, ndmin=2)
        self.targets = np.array(self.targets, ndmin=1)

    def get_X(self):
        return self.prototypes

    def get_y(self):
        return self.targets

    def get_hyper_rectangles(self):
        return self.hyper_rectangles

    def get_number_of_prototypes(self):
        return len(self.targets)
