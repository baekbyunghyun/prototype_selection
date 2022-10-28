#!/usr/bin/env python
"""
    Check the information on the data
    ---------------------------------
    (usage) dsdata.py fn n_per_C cols C - generate a problem with N X C points per class 
                               and C classes
            dsdata.py fn     - contour the given data with randomly selected
                               prototypes
"""

import sys
import numpy as np

from sklearn.datasets import load_svmlight_file


class ArgError(Exception):
    pass;


# define file format
SPARSE = 0
NORMAL = 1


def checkFormat(filename):
    file_format = NORMAL
    if filename.endswith(".spa"):  # deal with sparse data
        file_format = SPARSE
    return file_format


class DataInstances:
    def __init__(self, X=None, y=None):

        assert (X is not None and y is not None)

        # instance variable

        # shuffle X and y
        indices = np.random.randint(y.size, size=y.size)

        self.X = X[indices]
        self.y = y[indices]
        self.idx = 0

        self.rows = None
        self.cols = None
        self.nclass = None
        self.n_per_class = None
        self.classes = None

        self.setPara()

        self.oldmin = None
        self.oldmax = None

        self.normalized = False
        #self.scale()

    def setPara(self):
        self.rows, self.cols = self.X.shape
        self.resetClasses()
        self.classes = np.unique(self.y)
        self.nclass = len(self.classes)
        self.setNPerClass()

    def setNPerClass(self):
        self.n_per_class = np.zeros(self.nclass)
        for i, v in enumerate(self.classes):
            self.n_per_class[i] = len(np.where(self.y == v)[0])

    def resetClasses(self):
        import copy
        s = copy.deepcopy(self.y)
        target = np.unique(s)
        self.y = np.zeros(self.rows, dtype='int')
        for c, v in enumerate(target):
            ind = np.where(s == v)[0]
            self.y[ind] = c

    def saveInstances(self, filename=None):
        assert filename is not None, "File name is None!"

        # save pairs (y, X) into default.txt if a file name is not given.
        # Each column is delimited with ,.
        if checkFormat(filename) == SPARSE:
            from sklearn.datasets import dump_svmlight_file
            dump_svmlight_file(self.X, self.y, filename, zero_based=False)
        else:
            print('Filename : ', filename)
            data = np.column_stack((self.y, self.X))
            np.savetxt(filename, data, fmt="%s", delimiter=',')

    def scale(self):
        from sklearn.preprocessing import StandardScaler
        mu = np.mean(self.X, axis=0);
        var = np.var(self.X, axis=0);
        self.oldmin = np.min(self.X)
        self.oldmax = np.max(self.X)
        if not (all(mu) < sys.float_info.epsilon and all(var) <= 1):
            # X is not normalized
            scaler = StandardScaler(copy=True, with_mean=True, with_std=True);
            scaler.fit(self.X);
            self.X = scaler.fit_transform(self.X);
        self.normalized = True;

    def getXy(self):
        # return the instance array X and class array y
        return (self.X, self.y);

    def getIdx(self):
        return self.idx

    def getRows(self):
        # return the total of instances
        return self.rows;

    def getCols(self):
        return self.X.shape[1];

    def getNPerClass(self, i):
        # return the number of instances per class
        assert i <= 0 < self.nclass, "Out of range on class " + str(i)
        return self.n_per_cls[i];

    def getNoCls(self):
        # return the number of classes
        return self.nclass

    def getClasses(self):
        # return the array of  class targets
        return self.classes;

    def __str__(self):
        np.set_printoptions(suppress=True)
        msg = "------ Overview on data ------";
        msg += "\n data dimension    : {0} X {1}".format(self.rows, self.cols);
        msg += "\n no. of classes    : {0}".format(self.nclass);
        msg += "\n data size per cls : {0}".format(self.n_per_class);
        msg += "\n class labels      : {0}".format(self.classes);
        msg += "\n normalized        : {0}".format(self.normalized);
        msg += "\n min, max          : {0}, {1}".format(np.min(self.X), np.max(self.X));
        msg += "\n (old min, old max): ({0}, {1})".format(self.oldmin, self.oldmax);
        msg += "\n mean              : {0}".format(np.mean(self.X, axis=0));
        msg += "\n variance          : {0}".format(np.var(self.X, axis=0));
        msg += "\n recommend radius  : {0}".format(0.3 * np.abs(np.max(self.X) - np.min(self.X)));
        msg += "\n------------------------------";
        return msg;

    # implement an iterator

    def __len__(self):
        ' return the next value '
        return self.rows;

    def __getitem__(self, i):
        ' return the next value '
        return (i, self.y[i], self.X[i]);

    def __iter__(self):
        ' return the next value '
        self.idx = 0
        return self

    def __next__(self):
        ' return the next value '
        if 0 <= self.idx < self.rows:
            idx = self.idx
            self.idx = self.idx + 1
            return (idx, self.y[idx], self.X[idx])
        else:
            raise StopIteration()


class GenerateInstances:
    def __init__(self, filename=None, n_per_class=None, cols=None):

        assert filename is not None and n_per_class is not None and cols is not None, "Missing argument!"
        self.filename = filename
        self.n_per_class = n_per_class  # no. of data per class(list)
        self.nclass = len(self.n_per_class)  # no. of classes
        self.cols = cols
        self.rows = np.sum(self.n_per_class)  # the total of data
        self.classes = np.arange(self.nclass)  # a set of classes

        self.X = np.zeros((self.rows, self.cols))
        self.y = np.zeros(self.rows)

        self.gen2DInstances()

        self.dataObj = DataInstances(X=self.X, y=self.y)
        self.dataObj.saveInstances(filename=self.filename)

    def getDataObj(self):
        return self.dataObj

    def gen2DInstances(self):
        # class 0
        tot = self.n_per_class[0]
        count = 0
        self.y[count:tot] = self.classes[0]
        while count < tot:
            x = np.random.rand(self.cols)
            if not ((0.2 < x[0] < 0.8) and (0.2 < x[1] < 0.8)):
                self.X[count] = x
                count = count + 1
        print('count = ', count)

        # class 1
        tot = tot + self.n_per_class[1]
        self.y[count:tot] = self.classes[1]
        while count < tot:
            x = np.random.rand(self.cols)
            if (0.2 < x[0] < 0.8) and (0.2 < x[1] < 0.8) and (x[0] < x[1]):
                self.X[count] = x
                count = count + 1
        print('count = ', count)

        # class 2
        tot = tot + self.n_per_class[2]
        self.y[count:tot] = self.classes[2]
        while count < tot:
            x = np.random.rand(self.cols)
            if (0.2 < x[0] < 0.8) and (0.2 < x[1] < 0.8) and (x[0] > x[1]):
                self.X[count] = x
                count = count + 1
        print('count = ', count)


class ReadInstances:
    def __init__(self, filename=None):

        assert filename is not None, 'Specify a filename'

        self.filename = filename
        self.X = None
        self.y = None

        if checkFormat(self.filename) == SPARSE:
            self.loadSVMlightInstances()
        elif checkFormat(self.filename) == NORMAL:
            self.loadInstances()

        # instance variable
        self.dataObj = DataInstances(X=self.X, y=self.y)

    def getDataObj(self):
        return self.dataObj

    def loadInstances(self):
        # read X and y from a CSV filename in CSV format
        XX = np.loadtxt(self.filename, delimiter=',')  # ,-delimited
        self.rows, self.cols = XX.shape
        self.X = XX[:, 1:self.cols]
        self.y = XX[:, 0]

    def loadSVMlightInstances(self):
        # read X and Y from a filename with sparse format
        from scipy.sparse import coo_matrix
        XX, self.y = load_svmlight_file(self.filename)


        self.X = np.array(coo_matrix(XX, dtype=np.float).todense())
