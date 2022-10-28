#!/usr/bin/env python

"""
   Select the set of prototypes using DSHwang or Bien's algo.
   ----------------------------------------------------------
   (DSHwang) dsneardata.py data.txt
   (Bien)    dsneardata.py data.txt radius
"""

import numpy as np

import sklearn.metrics.pairwise as sk

from ds_paper.dsdata import *


class NearData:
    def __init__(self, X, Y):
        assert(X is not None and Y is not None)

        # set X and Y
        self.X = X
        self.Y = Y
        self.N, self.M = X.shape
        
        # get the no. of classes
        self.K = np.unique(self.Y)

        # compute euclidean distance
        # metric : l2, l1, manhattan, cityblock
        self.dist = sk.pairwise_distances(X, metric='l2')
        #print "dist size: ", self.dist.shape

    def getDist(self, i=None):
        if i is None:
           return self.dist;
        elif isinstance(i, int):
           return self.dist[i];

    def getNoOfCls(self):
        return self.K;

    def getPrototype(self):
        raise NotImplementedError('No method in class %s' % self.__class__.__name__);

    def __str__(self):
        s = 'NearData: %d X %d' % ( self.N, self.M);
        return s;

    def getRows(self):
        return self.N

    def getCols(self):
        return self.M
        
    def getA(self):
        pass
    
class NearSameData(NearData):
    def __init__(self, X, Y, theta=50):

        NearData.__init__(self, X, Y);

        # set the rate of line segment
        self.rate_seg2 = 100.0 - theta*np.ones(X.shape[1])
        self.rate_seg1 = theta*np.ones(X.shape[1])

        # compute indices per data within ep
        self.nn, self.ep = self.nearest();

        # compute the no. of same class
        self.cls_count = np.zeros(self.N);
        self.setClassInfo();

        # compute a coefficient matrix A
        self.setA();

    def nearest(self):
        nn = list(); 
        epsilon = list();
        for i in np.arange(self.N):
            tmp_near = list(); # the same class
            tmp_near.append(i);
            vals = self.dist[i];
            indices = np.argsort(vals);
            first = indices[0]; # should be i
            for j in np.arange(1,self.N):
                second = indices[j];
                if not np.allclose(self.Y[i],self.Y[second]):
                   # found the first example with the different class
                   break;
                else:
                   tmp_near.append(second);
                   first = second;

            # compute the midpoint of X[1] and X[2]
            # check the radius rate
            mid = 0.01 * (self.rate_seg2*self.X[second]+self.rate_seg1*self.X[first])
            radius = float(sk.euclidean_distances( self.X[i].reshape(1,-1), mid.reshape(1,-1) ));

            near = list();
            for it in tmp_near:
                # confirm that nearest items is within the radius
                if self.dist[i][it] < radius:
                   near.append( it );

            # add the set of nearest points & the real radius
            #    after distance adaptation
            nn.append(near);
            epsilon.append(float(radius));

        return nn, epsilon;

    def setClassInfo(self):
        for i in np.arange(self.N):
            self.cls_count[i] = len(self.nn[i]);

    def setA(self):
        # set the coefficient matrix A_ij
        A = np.zeros((self.N, self.N));
        for i in np.arange(0,self.N):
            for j in np.arange(0,self.N):
                if i in self.nn[j]:
                   A[i,j] = 1;
        #print ">>>> gendata: ", A.shape
        return A;

    def getIndex(self,i=None):
        if i is not None:
            return self.nn[i];
        else:
            return self.nn;

    def getEp(self, i=None):
        if i is not None:
            return self.ep[i];
        else:
            return self.ep;

    def getClassInfo(self):
        return self.cls_count;

    def getPrototype(self):
        L = list();
        check = np.zeros(self.N, dtype='int64');
        indices = np.argsort(self.cls_count);
        #print 'cls = ', self.cls_count[indices[::-1]];
        #print 'index = ', indices[::-1];
        cnt = 0;
        for ea in indices[::-1]:

            #print '%2d: %2d %s %.2f'%(ea, self.cls_count[ea], self.nn[ea],self.epsilon[ea])

            if np.sum(check) == self.N:
               break;
            elif np.sum(check[self.nn[ea]]) != self.cls_count[ea]:
               check[self.nn[ea]] = 1;
               L.append(ea);
               cnt = cnt + 1;
               #print " ... *"

        #print ">>> ", cnt, " protos"
        #print check, " "
        return L;

    def __str__(self):
        s1 = NearData.__str__(self);
        s2 = '\n\tNearSameData: ep = %s' % ( str(self.ep));
        return s1 + s2;

class NearDataWithEp(NearData):
    def __init__(self, X, Y, ep = 1.0):
        assert( ep is not None );

        NearData.__init__(self, X, Y);
        self.ep = ep

        # compute indices per data within ep
        self.nn = self.nearest();

        # compute the no. of same class and anti class, 0-th index for total
        self.cls_count = np.zeros((self.N, 3));
        self.setClassInfo();
        self.setA(); # compute a coefficient matrix A

    def nearest(self):
        nn = list();
        for i in np.arange(self.N):
            near = np.where(self.dist[i]<self.ep)[0];
            nn.append(near);
        return nn;

    def setClassInfo(self):
        for i in np.arange(self.N):
            # find the nearest indices of X[i]
            # check if Y[i] == Y[within]
            same_cls = np.where(self.Y[self.nn[i]]==self.Y[i])[0].size;
            other_cls = self.nn[i].size - same_cls;
            #print 'i=',i,'all=', nearest, ' same= ', nearest[same_cls], \
            #          ' other=', nearest[other_cls]

            self.cls_count[i,0] = self.nn[i].size;
            self.cls_count[i,1] = same_cls;
            self.cls_count[i,2] = other_cls;

    def setA(self):
        # set the coefficient matrix A_ij
        A = np.zeros((self.N, self.M));
        for i in np.arange(0,self.N):
            for j in np.arange(0,self.M):
                if i in self.nn[j]:
                   A[i,j] = 1;
        return A;

    def getIndex(self,i=None):
        if i is not None:
            return self.nn[i];
        else:
            return self.nn;

    def getEp(self, i=None):
        return self.Ep;

    def getClassInfo(self):
        return self.cls_count;

    def getPrototype(self):
        L = list();
        check = np.zeros(self.N, dtype='int64');
        indices = np.argsort(self.cls_count[:,1]);
        cnt = 0;
        for ea in indices[::-1]:

            if np.sum(check) == self.N:
               break;
            elif np.sum(check[self.nn[ea]]) < self.cls_count[ea,1]:
               check[self.nn[ea]] = 1;
               L.append(ea);
               cnt = cnt + 1;

        return L;

    def __str__(self):
        s1 = NearData.__str__(self);
        s2 = '\n\tNearDataWithEp: ep = %f' % ( self.ep );
        return s1 + s2;

class NearSameDataWithEp(NearData):
    def __init__(self, X, Y, ep = 1.0):

        assert( ep is not None );

        NearData.__init__(self, X, Y);
        self.Ep = np.ones(self.N)*ep

        # compute indices per data within ep
        self.nn, self.Ep = self.nearest();

        # compute the no. of same class and anti class, 0-th index for total
        self.cls_count = np.zeros(self.N);
        self.setClassInfo();
        self.setA(); # compute a coefficient matrix A

    def nearest(self):
        nn = list(); 
        epsilon = self.Ep;
        for i in np.arange(self.N):
            tmp_near = list(); # the same class
            tmp_near.append(i);
            vals = self.dist[i];
            indices = np.argsort(vals);
            first = indices[0]; # should be i
            for j in np.arange(1,self.N):
                second = indices[j];
                if not np.allclose(self.Y[i],self.Y[second]) and vals[second] < self.Ep[i]:
                   # found the first example with the different class
                   # compute the midpoint of X[1] and X[2]
                   # check the radius rate
                   mid = ( self.X[first] + self.X[second]) / 2.0;
                   radius = sk.euclidean_distances( self.X[first].reshape(1,-1), mid.reshape(1,-1) )*1.10;
                   self.Ep[i] = radius
                   break;

                else:
                   tmp_near.append(second);
                   first = second;

            near = list();
            for it in tmp_near:
                # confirm that nearest items is within the radius
                if self.dist[i][it] < self.Ep[i]:
                   near.append( it );

            # add the set of nearest points & the real radius
            #    after distance adaptation
            nn.append(near);

        return nn, self.Ep;

    def setClassInfo(self):
        for i in np.arange(self.N):
            self.cls_count[i] = len(self.nn[i]);

    def setA(self):
        # set the coefficient matrix A_ij
        A = np.zeros((self.N, self.N));
        for i in np.arange(0,self.N):
            for j in np.arange(0,self.N):
                if i in self.nn[j]:
                   A[i,j] = 1;
        return A;

    def getIndex(self,i=None):
        if i is not None:
            return self.nn[i];
        else:
            return self.nn;

    def getEp(self, i=None):
        if i is not None:
            return self.Ep[i]
        else:
            return self.Ep;

    def getClassInfo(self):
        return self.cls_count;

    def getPrototype(self):
        L = list();
        check = np.zeros(self.N, dtype='int64');
        indices = np.argsort(self.cls_count);
        cnt = 0;
        for ea in indices[::-1]:

            if np.sum(check) == self.N:
               break;
            elif np.sum(check[self.nn[ea]]) < self.cls_count[ea]:
               check[self.nn[ea]] = 1;
               L.append(ea);
               cnt = cnt + 1;

        return L;

    def __str__(self):
        s1 = NearData.__str__(self);
        s2 = '\n\tNearSameDataWithEp: ep = %f' % ( self.ep[0] );
        return s1 + s2;
