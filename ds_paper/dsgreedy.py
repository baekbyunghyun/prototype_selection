#!/usr/bin/python

"""
    Select a set of prototypes using Bien's and DSHwang's algorithms
    ---------------------------------------------------------------
    (Bien) dsgreedy.py bien data.txt radius
    (DSHwang) dsgreedy.py mine data.txt tau theta
              dsgreedy.py mixed data.txt tau radius 
"""

import numpy as np
import time

from ds_paper.dsneardata import *


class SCP:
    def __init__(self, Dxz, y):
        self.Dxz = Dxz
        self.y = y
        self.n, self.m = self.Dxz.shape
        self.classes = np.unique(np.array(y, 'int'))
        self.K = self.classes.size

        self.protos = list()
        self.exec_time = None

    def get_prototype(self):
        return self.protos

    def get_number_of_prototypes(self):
        return len(self.protos)

    def get_rate(self):
        return 1.0 * len(self.protos) / self.n * 100.0

    def get_elapsed_time(self):
        return self.exec_time

    def get_ep(self):
        raise NotImplemented

    def get_each_ep(self, i):
        raise NotImplemented

    def greedy(self):
        raise NotImplemented

    def __str__(self):
        np.set_printoptions(suppress=True)
        msg = "\n------ Results ------"
        msg += "\n no. of data       : {0}".format(self.n)
        msg += "\n no. of classes    : {0}".format(self.K)
        msg += "\n no. of pototypes  : {0}".format(len(self.protos))
        msg += "\n selection rate    : {0}".format(self.get_rate())
        msg += "\n execution time    : {0}".format(self.get_elapsed_time())
        msg += "\n---------------------"
        return msg


class ElasticRadii(SCP):
    def __init__(self, X, y, tau=0, theta=50):

        assert (X is not None and y is not None)

        nsd = NearSameData(X, y, theta=theta)
        self.Dxz = nsd.getDist()

        SCP.__init__(self, self.Dxz, y)
        self.ep = nsd.getEp()  # radii array of each X[i]
        self.tau = tau

        # define necessary data structure
        self.C = np.zeros((self.n, self.K))
        for cl in self.classes:
            self.C[:, cl] = (y == cl)

        self.B = np.zeros((self.n, self.n));
        self.has = np.zeros(self.n)
        for i in np.arange(self.n):
            self.B[i, :] = self.Dxz[i, :] <= self.ep[i];
            self.has[i] = np.sum(self.B[i, :])

        self.covered = np.zeros(self.n, dtype='int64');

        # check whether a self point is included or not
        # self.scores = np.dot(self.B, self.C)-1; # not including a self point
        self.scores = np.dot(self.B, self.C);  # including a self point

    def greedy(self):
        start = time.clock()

        every = list(np.arange(self.n))
        covered = np.zeros(self.n)

        # select prototypes
        i = 0  # check the number of iterations
        while (np.sum(covered) != self.n):
            max_val = np.max(self.has[every])
            ind = 0
            for each in every:
                if self.has[each] == max_val:
                    ind = each
                    break
            candidate = ind

            self.protos.append(candidate)
            ind = np.where(self.B[candidate, :] == 1)[0]
            for each in ind:
                if each in every:
                    every.remove(each)
            covered[ind] = 1
            i = i + 1;

        # prune prototypes with tau
        excluded = np.where(self.has < self.tau)[0]
        for each in excluded:
            if each in self.protos:
                self.protos.remove(each)
        self.protos = np.unique(self.protos)

        self.exec_time = time.clock() - start

        return self.protos

    def get_ptototypes(self):
        return self.protos

    def get_ep(self):
        return self.ep;


class ElasticRadius(SCP):
    def __init__(self, X, y, tau=1, ep=.5):

        assert (X is not None and y is not None)

        nsd = NearSameDataWithEp(X, y, ep=ep)
        self.Dxz = nsd.getDist()

        SCP.__init__(self, self.Dxz, y)
        self.ep = nsd.getEp()  # radii array of each X[i]
        self.tau = 1

        # define necessary data structure
        self.C = np.zeros((self.n, self.K))
        for cl in self.classes:
            self.C[:, cl] = (y == cl)

        self.B = np.zeros((self.n, self.n));
        self.has = np.zeros(self.n)
        for i in np.arange(self.n):
            self.B[i, :] = self.Dxz[i, :] <= self.ep[i];
            self.has[i] = np.sum(self.B[i, :])

        self.covered = np.zeros(self.n, dtype='int64');

        # check whether a self point is included or not
        # self.scores = np.dot(self.B, self.C)-1; # not including a self point
        self.scores = np.dot(self.B, self.C);  # including a self point

    def greedy(self):
        start = time.clock()

        every = list(np.arange(self.n))
        covered = np.zeros(self.n)

        # select prototypes
        i = 0  # check the number of iterations
        while (np.sum(covered) != self.n):
            max_val = np.max(self.has[every])
            ind = 0
            for each in every:
                if self.has[each] == max_val:
                    ind = each
                    break
            candidate = ind

            self.protos.append(candidate)
            ind = np.where(self.B[candidate, :] == 1)[0]
            for each in ind:
                if each in every:
                    every.remove(each)
            covered[ind] = 1
            i = i + 1;

        # prune prototypes with tau
        excluded = np.where(self.has < self.tau)[0]
        for each in excluded:
            if each in self.protos:
                self.protos.remove(each)
        self.protos = np.unique(self.protos)

        self.exec_time = time.clock() - start

        return self.protos

    def get_prototypes(self):
        return self.protos

    def get_ep(self):
        return self.ep

    def get_each_ep(self, i):
        return self.ep[i]


class FixedRadius(SCP):
    def __init__(self, X, y, ep=.5):
        assert (X is not None and y is not None);
        self.ep = ep
        nsd = NearDataWithEp(X, y, ep=self.ep)
        self.Dxz = nsd.getDist()

        SCP.__init__(self, self.Dxz, y)

        # define necessary data structure
        self.lambda2 = 1.0 / self.y.size
        self.C = np.zeros((self.n, self.K))
        for cl in self.classes:
            self.C[:, cl] = (y == cl)

        W = 2 * self.C - 1
        self.covered = np.zeros(self.n, dtype='int64');
        self.B = self.Dxz < self.ep
        self.scores = np.dot(self.B, self.C) - 1  # including a self point
        self.ncovered = list()

    def greedy(self):
        start = time.clock()

        i = 0
        while (True):
            i = i + 1;
            iimax = np.argmax(self.scores);
            pmax = int(np.floor(iimax / self.K));
            kmax = int(iimax - pmax * self.K);

            if self.scores[pmax, kmax] >= self.lambda2:

                # add this prototype increases objective by the most
                self.protos.append(pmax);

                # update scores

                # identify points that are no longer uncovered
                c_val = np.where(self.covered, False, True);
                justcovered = np.zeros(self.n, dtype='int64');
                for i in np.arange(self.n):
                    if self.B[i, pmax] == True and \
                                    self.C[i, kmax] == True and c_val[i] == True:
                        justcovered[i] = 1;

                self.covered = self.covered + justcovered;

                # determine which prototypes's scores were affected
                mval = np.sum(self.B[:, justcovered], axis=0);
                affectedprotos = np.where(mval > 0)[0];

                c_val = np.where(self.C[:, kmax], False, True);
                cnt = 0;
                not_incl = list();
                for i in np.arange(self.n):
                    if self.B[i, pmax] == True and \
                                    c_val[i] == True:
                        not_incl.append(i);
                        cnt = cnt + 1;
                # val = [np.sum(justcovered), np.sum(self.B[:,pmax] == c_val)]
                if cnt > 0:
                    val = [np.sum(justcovered), cnt, not_incl];
                    self.ncovered.append(val);

                # for each affected prototype, determine the no. of points
                # that it can cover that no longer need to be covered
                reduce = [np.sum(self.B[:, p] * justcovered) for p in affectedprotos];
                self.scores[affectedprotos, kmax] = self.scores[affectedprotos, kmax] - reduce;

            else:
                break;

        self.protos = np.unique(self.protos)

        self.exec_time = time.clock() - start

        return self.protos

    def get_prototypes(self):
        return self.protos

    def get_ep(self):
        return self.ep


class RandomSphere(SCP):
    def __init__(self, X, y, tau=1):
        assert (X is not None and y is not None);

        nsd = NearSameData(X, y, theta=100);
        self.Dxz = nsd.getDist();
        SCP.__init__(self, self.Dxz, y)
        self.ep = nsd.getEp();  # radii array of each X[i]'s

        self.tau = tau  # tau parameter
        self.has = np.zeros(self.n)  # no. of covered points

        # check the members of a sphere
        self.B = np.zeros((self.n, self.n));
        for i in np.arange(self.n):
            self.B[i, :] = self.Dxz[i, :] <= self.ep[i];
            self.has[i] = np.sum(self.B[i, :])

    def greedy(self):
        start = time.clock()

        every = list(np.arange(self.n))
        covered = np.zeros(self.n)

        # select prototypes
        i = 0  # check the number of iterations
        while (np.sum(covered) != self.n):
            candidate = np.random.choice(every)
            self.protos.append(candidate)
            ind = np.where(self.B[candidate] == 1)[0]
            for each in ind:
                if each in every:
                    every.remove(each)
            covered[ind] = 1
            i = i + 1;

        # prune prototypes with tau
        excluded = np.where(self.has < self.tau)[0]
        for each in excluded:
            if each in self.protos:
                self.protos.remove(each)

        self.protos = np.unique(self.protos)

        self.exec_time = time.clock() - start

        return self.protos

    def get_prototypes(self):
        return self.protos

    def get_ep(self):
        return self.ep


class GreedySphere(SCP):
    def __init__(self, X, y):
        assert (X is not None and y is not None);

        nsd = NearSameData(X, y, theta=100);
        self.Dxz = nsd.getDist();
        SCP.__init__(self, self.Dxz, y)
        self.ep = nsd.getEp();  # radii array of each X[i]'s

        self.has = np.zeros(self.n)  # no. of covered points

        # check the members of a sphere
        self.B = np.zeros((self.n, self.n));
        for i in np.arange(self.n):
            self.B[i, :] = self.Dxz[i, :] <= self.ep[i];
            self.has[i] = np.sum(self.B[i, :])

    def greedy(self):
        start = time.clock()

        covered = np.zeros(self.n)

        # select prototypes
        i = 0  # check the number of iterations
        while (np.sum(covered) != self.n):
            candidate = np.argmax(self.has)
            self.protos.append(candidate)
            ind = np.where(self.B[candidate] == 1)[0]
            if len(ind) is 0:
                print('B = ', self.B[candidate])
                print('candidate = ', candidate)
                print('ind = ', ind)
            for each in ind:
                self.has[each] = self.has[each] - 1
            covered[ind] = 1
            i = i + 1;

        self.protos = np.unique(self.protos)

        self.exec_time = time.clock() - start

        return self.protos

    def get_prototypes(self):
        return self.protos

    def get_ep(self):
        return self.ep

