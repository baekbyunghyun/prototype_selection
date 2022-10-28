import numpy as np


class PlotInstance:
    def __init__(self, X=None, y=None):
        assert X is not None and y is not None, 'Invalid argument.'

        self.X = X
        self.y = y
        self.classes = np.unique(self.y)
        self.K = len(self.classes)
        self.N = len(self.y) / len(self.classes)

        self.x_gap = np.ptp(self.X[:, 0]) / 50.0
        self.y_gap = np.ptp(self.X[:, 1]) / 50.0

        self.x_min = self.X[:, 0].min()
        self.x_max = self.X[:, 0].max()
        self.y_min = self.X[:, 1].min()
        self.y_max = self.X[:, 1].max()

        if np.abs(self.x_max - self.x_min) > np.abs(self.y_max - self.x_min):
            self.axis = [self.x_min - self.x_gap,
                         self.x_max + self.x_gap,
                         self.x_min - self.y_gap,
                         self.x_max + self.y_gap]
        else:
            self.axis = [self.y_min - self.x_gap,
                         self.y_max + self.x_gap,
                         self.y_min - self.y_gap,
                         self.y_max + self.y_gap]
