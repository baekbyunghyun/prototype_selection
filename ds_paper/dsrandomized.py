#!/usr/bin/python

'''
   randomized rounding by J. Bien with postprocessing
'''

from ds_paper.dsplot import *
from .dsneardata import NearDataWithEp


class RandomizedScp:
    def __init__(self, y, nn, cls_count ):
        self.y = np.array(y,'int'); self.nn = nn;
        self.cls_count = cls_count;

        self.N = self.y.shape[0]; # no. of samples
        self.Cls = np.unique(self.y); # class labels
        self.Chi = np.arange(self.N); # sample indices

        self.not_in = list(); # set(Chi - Chi_l)
        self.Chi_l = list();
        self.protos = list();
        for c in self.Cls:
            chi = np.where(self.y==c)[0];
            chi = np.array(chi);
            temp = np.setdiff1d(self.Chi, chi);

            self.Chi_l.append( chi );
            self.not_in.append(np.array(temp));
            self.protos.append(list());
        self.delta_eta();

    def delta_xi(self, j):
        idx_cls = np.where( self.Cls == self.y[j])[0];
        added = list();
        for i in self.protos[idx_cls]:
            added = np.union1d(added, self.nn[i]);

        unadded = np.setdiff1d(self.nn[j], added);
        xi = np.intersect1d(self.Chi_l[idx_cls], unadded);
        return xi.size;

    def delta_eta(self):
        self.eta = list();
        for j in self.Chi:
            label = int(self.y[j]);
            temp = np.intersect1d(self.nn[j],self.not_in[label]);
            self.eta.append(temp);

    def getPrototype(self):
        L = [ ea for S in self.protos for ea in S ]
        return L 

    def greedy(self):
        covered = [False for ea in self.Chi];
        pcount = 0;
        #lambda2 = 1.0/self.N;
        while True:
            obj = np.zeros(self.N);
            for j in self.Chi:
                #obj[j] = self.delta_xi(j) - len(self.eta[j]);
                obj[j] = self.delta_xi(j);
#                print 'obj[',j,']= ', obj[j], ' xi=', self.delta_xi(j), ' eta=', len(self.eta[j]);

            val = np.max( obj ); idx = np.where(obj==val)[0];
            for ii in idx:
               if covered[ii]==False:
                  cls = int(self.y[ii]);
                  self.protos[cls].append(ii); pcount+=1;
            else:
               break;

            print(zip(self.Chi,  self.protos));

        # postprocess unadded data
        acount = 0;
        for ea in self.Chi:
            if covered[ea]==False:
               cls = int(self.y[ea]);
               self.protos[cls].append(ea); acount+=1;
               for k in self.nn[ea]:
                   covered[k] = True;
        #print pcount, ' protos', acount, ' added'
        return self.protos;

if __name__ == "__main__":
     dd = ReadInstances(filename=sys.argv[1]); # data filename
     di = dd.getDataObj()
     radius = float(sys.argv[2]); # be radius

     X, y = di.getXy(); # get the training data

     ep = NearDataWithEp(X, y, radius); # get the object of computing neighbors
     nearest = ep.getIndex(); # compute neighbors with ep
     cls_count = ep.getClassInfo();

     gr = RandomizedScp( y, nearest, cls_count );
     P = gr.greedy();
     print('P: ', P)
     L = [ea for S in P for ea in S];
     print('No. of selected prototypes :', len(L));

     dp = PlotInstances(X, y); # get the ploting object
     dp.plotWithRadius(L, radius);
     dp.draw_graph()


