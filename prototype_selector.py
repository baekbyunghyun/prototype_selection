import os

from dataset.data_reader import DataReader
from dataset.dataset import Dataset
from ds_paper.dsgreedy import *
from ds_paper.dsrandomized import *
from hyper_rectangle.hyper_rectangle_v10 import *
from hyper_rectangle.hyper_rectangle_v20 import *
from hyper_rectangle.hyper_rectangle_v30 import *
from hyper_rectangle.hyper_rectangle_v40 import *


class PrototypeSelector:
    TYPE_ALGORITHM_RSC = 'RSC'
    TYPE_ALGORITHM_GSC = 'GSC'
    TYPE_ALGORITHM_IPS = 'IPS'
    TYPE_ALGORITHM_RR = 'RR'
    TYPE_ALGORITHM_MIXED = 'MIXED'
    TYPE_ALGORITHM_PBL = 'PBL'
    TYPE_ALGORITHM_HRPS_V_00 = 'HRPS_V_00'
    TYPE_ALGORITHM_HRPS_V_10 = 'HRPS_V_10'
    TYPE_ALGORITHM_HRPS_V_20 = 'HRPS_V_20'
    TYPE_ALGORITHM_HRPS_V_30 = 'HRPS_V_30'
    TYPE_ALGORITHM_HRPS_V_40 = 'HRPS_V_40'

    def __init__(self, algorithm=None, X=None, y=None,
                 parameter_k=1, parameter_tau=1.0, parameter_theta=50.0, parameter_radius=0.5, debug=False):
        assert algorithm is not None and X is not None and y is not None, 'Invalid prototype selection Info.'

        self.algorithm = algorithm
        self.X = X
        self.y = y
        self.parameter_k = parameter_k
        self.parameter_tau = parameter_tau
        self.parameter_theta = parameter_theta
        self.parameter_radius = parameter_radius
        self.debug = debug

        self.prototype_selector = None
        self.X_prototype = None
        self.y_prototype = None

    def select(self):
        is_rect = False

        if self.algorithm == PrototypeSelector.TYPE_ALGORITHM_RSC:
            self.prototype_selector = RandomSphere(self.X, self.y, tau=self.parameter_tau)
            self.prototype_selector.greedy()

        elif self.algorithm == PrototypeSelector.TYPE_ALGORITHM_GSC:
            self.prototype_selector = GreedySphere(self.X, self.y)
            self.prototype_selector.greedy()

        elif self.algorithm == PrototypeSelector.TYPE_ALGORITHM_IPS:
            self.prototype_selector = FixedRadius(self.X, self.y, ep=self.parameter_radius)
            self.prototype_selector.greedy()

        elif self.algorithm == PrototypeSelector.TYPE_ALGORITHM_RR:
            near_data = NearDataWithEp(self.X, self.y, ep=self.parameter_radius)

            near_data_index = near_data.getIndex()
            class_count = near_data.getClassInfo()

            self.prototype_selector = RandomizedScp(self.y, near_data_index, class_count)
            self.prototype_selector.greedy()

        elif self.algorithm == PrototypeSelector.TYPE_ALGORITHM_MIXED:
            self.prototype_selector = ElasticRadius(self.X, self.y, tau=self.parameter_tau, ep=self.parameter_radius)
            self.prototype_selector.greedy()

        elif self.algorithm == PrototypeSelector.TYPE_ALGORITHM_PBL:
            self.prototype_selector = ElasticRadii(self.X, self.y, tau=self.parameter_tau, theta=self.parameter_theta)
            self.prototype_selector.greedy()

        elif self.algorithm == PrototypeSelector.TYPE_ALGORITHM_HRPS_V_10:
            is_rect = True

            self.prototype_selector = HyperRectangleSelector(X=self.X, y=self.y, tau=self.parameter_tau)
            self.prototype_selector.greedy()

        elif self.algorithm == PrototypeSelector.TYPE_ALGORITHM_HRPS_V_20:
            is_rect = True

            self.prototype_selector = HyperRectangleSelectorV20(X=self.X, y=self.y, tau=self.parameter_tau, debug=self.debug)
            self.prototype_selector.greedy()

        elif self.algorithm == PrototypeSelector.TYPE_ALGORITHM_HRPS_V_30:
            is_rect = True

            self.prototype_selector = HyperRectangleSelectorV30(X=self.X, y=self.y, tau=self.parameter_tau)
            self.prototype_selector.greedy()

        elif self.algorithm == PrototypeSelector.TYPE_ALGORITHM_HRPS_V_40:
            is_rect = True

            self.prototype_selector = HyperRectangleSelectorV40(X=self.X, y=self.y, tau=self.parameter_tau)
            self.prototype_selector.greedy()

        else:
            raise Exception("Invalid prototype algorithm")

        if is_rect:
            self.X_prototype = self.prototype_selector.get_X()
            self.y_prototype = self.prototype_selector.get_y()

        else:
            self.X_prototype = self.X[self.prototype_selector.get_prototypes()]
            self.y_prototype = self.y[self.prototype_selector.get_prototypes()]

        return self.X_prototype, self.y_prototype

    def get_ep(self):
        if ((self.algorithm == PrototypeSelector.TYPE_ALGORITHM_IPS)
                or (self.algorithm == PrototypeSelector.TYPE_ALGORITHM_RR)):
            return [self.parameter_radius for _ in range(self.X_prototype.shape[0])]

        return self.prototype_selector.get_ep()

    def get_hyper_rectangles(self):
        return self.prototype_selector.get_hyper_rectangles()

    def get_num_of_prototypes(self):
        return len(self.X_prototype)

    def get_prototype_index(self):
        return self.prototype_selector.get_prototypes()

    def get_hyper_sphere_radius(self):
        return self.prototype_selector.get_ep()

    @staticmethod
    def is_valid_algorithm(prototype_algorithm):
        if prototype_algorithm == PrototypeSelector.TYPE_ALGORITHM_RSC:
            return True

        if prototype_algorithm == PrototypeSelector.TYPE_ALGORITHM_GSC:
            return True

        if prototype_algorithm == PrototypeSelector.TYPE_ALGORITHM_IPS:
            return True

        if prototype_algorithm == PrototypeSelector.TYPE_ALGORITHM_RR:
            return True

        if prototype_algorithm == PrototypeSelector.TYPE_ALGORITHM_MIXED:
            return True

        if prototype_algorithm == PrototypeSelector.TYPE_ALGORITHM_PBL:
            return True

        if prototype_algorithm == PrototypeSelector.TYPE_ALGORITHM_HRPS_V_00:
            return True

        return False


ROOT_DIR_PATH = r'D:\python_workspace\workspace\prototype_selection\feature'
SRC_FILE_NAME = 'entropy_histogram_l6.spa'
TYPE_OF_PROTOTYPE_SELECTION = PrototypeSelector.TYPE_ALGORITHM_HRPS_V_20


if __name__ == '__main__':
    parameter_theta_list = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]

    for theta in parameter_theta_list:
        dataset = DataReader(os.path.join(ROOT_DIR_PATH, SRC_FILE_NAME)).get_dataset()
        dataset.scale()
        dataset.shuffle()

        print(dataset)

        selector = PrototypeSelector(algorithm=TYPE_OF_PROTOTYPE_SELECTION, X=dataset.X, y=dataset.y, parameter_tau=theta, debug=True)
        X_prototype, y_prototype = selector.select()

        prototype_dataset = Dataset(X=X_prototype, y=y_prototype)
        prototype_dataset.save_dataset(os.path.join(ROOT_DIR_PATH, 'prototype_{0}_{1}.spa'.format(SRC_FILE_NAME, theta)))
