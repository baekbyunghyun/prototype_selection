from ds_paper.dsgreedy import *
from ds_paper.dsrandomized import *
from hyper_rectangle.hyper_rectangle_v10 import *


class Prototype:
    ALGORITHM_RSC = 'RSC'
    ALGORITHM_GSC = 'GSC'
    ALGORITHM_IPS = 'IPS'
    ALGORITHM_RR = 'RR'
    ALGORITHM_MIXED = 'MIXED'
    ALGORITHM_PBL = 'PBL'
    ALGORITHM_HRPS = 'HRPS'

    def __init__(self, algorithm=None, parameter_k=1, parameter_tau=1, parameter_theta=50, parameter_radius=0.5):
        self.algorithm = algorithm
        self.parameter_k = parameter_k
        self.parameter_tau = parameter_tau
        self.parameter_theta = parameter_theta
        self.parameter_radius = parameter_radius

        self.prototype_selector = None

        self.X = None
        self.y = None

        self.is_rect = False
        self.num_of_prototypes = 0

    @staticmethod
    def is_valid_algorithm(prototype_algorithm):
        if prototype_algorithm == Prototype.ALGORITHM_RSC:
            return True

        if prototype_algorithm == Prototype.ALGORITHM_GSC:
            return True

        if prototype_algorithm == Prototype.ALGORITHM_IPS:
            return True

        if prototype_algorithm == Prototype.ALGORITHM_RR:
            return True

        if prototype_algorithm == Prototype.ALGORITHM_MIXED:
            return True

        if prototype_algorithm == Prototype.ALGORITHM_PBL:
            return True

        if prototype_algorithm == Prototype.ALGORITHM_HRPS:
            return True

        return False

    def select(self, X, y):
        self.is_rect = False

        if self.algorithm == Prototype.ALGORITHM_RSC:
            self.prototype_selector = RandomSphere(X, y, tau=self.parameter_tau)

        elif self.algorithm == Prototype.ALGORITHM_GSC:
            self.prototype_selector = GreedySphere(X, y)

        elif self.algorithm == Prototype.ALGORITHM_IPS:
            self.prototype_selector = FixedRadius(X, y, ep=self.parameter_radius)

        elif self.algorithm == Prototype.ALGORITHM_RR:
            near_data = NearDataWithEp(X, y, ep=self.parameter_radius)

            nearest = near_data.getIndex()
            class_count = near_data.getClassInfo()

            self.prototype_selector = RandomizedScp(y, nearest, class_count)

        elif self.algorithm == Prototype.ALGORITHM_MIXED:
            self.prototype_selector = ElasticRadius(X, y, tau=self.parameter_tau, ep=self.parameter_radius)

        elif self.algorithm == Prototype.ALGORITHM_PBL:
            self.prototype_selector = ElasticRadii(X, y, tau=self.parameter_tau, theta=self.parameter_theta)

        elif self.algorithm == Prototype.ALGORITHM_HRPS:
            self.is_rect = True

            self.prototype_selector = HyperRectangleSelector(X=X, y=y, tau=self.parameter_tau)

        else:
            print("[ERROR] Invalid prototype algorithm. ", self.algorithm)

            return X, y

        hr = self.prototype_selector.greedy()

        if self.is_rect:
            self.num_of_prototypes = self.prototype_selector.get_number_of_prototypes()

            X_prototype = self.prototype_selector.get_X()
            y_prototype = self.prototype_selector.get_y()

            self.ep = hr

        else:
            prototype_index = self.prototype_selector.get_prototypes()
            self.num_of_prototypes = self.prototype_selector.get_number_of_prototypes()

            X_prototype = X[prototype_index]
            y_prototype = y[prototype_index]

            if (self.algorithm == Prototype.ALGORITHM_IPS) or (self.algorithm == Prototype.ALGORITHM_RR):
                self.ep = [self.parameter_radius for i in range(X_prototype.shape[0])]

            else:
                self.ep = self.prototype_selector.get_ep()

        return X_prototype, y_prototype
