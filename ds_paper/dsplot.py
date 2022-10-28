#!/usr/bin/env python
"""
    Plot the data with picked points
    ---------------------------------
    (usage) dsplot.py fn
"""
import matplotlib.pyplot as plt
from ds_paper.dsdata import *


def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
    """
    Minimize chartjunk by stripping out unnecesasry contour borders and axis ticks
    
    The top/right/left/bottom keywords toggle whether the corresponding contour border is drawn
    """
    ax = axes or plt.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)
    
    #turn off all ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    
    #now re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()

def stacked_bar_graph(k, bias1, var1, bias2, var2, prob = "Tested"):
    n = len(bias1);
    ind = np.arange(n)
    ymax = np.max(np.vstack((bias1,bias2)))+.1
    xlabel = [ str(i) for i in np.arange(1, k+1, 2)];
    red, blue = '#B2182B', '#2166AC'
    width = 1.0
    
    plt.subplot(121)
    plt.bar(ind, bias1, color = red, label = 'Bias', width=width)
    plt.bar(ind, var1, bottom=bias1, color = blue, label='Var', width=width)
    plt.xticks(ind+.5, xlabel, rotation='horizontal')
    plt.xlabel('K')
    plt.ylabel('Error')
    plt.axis([0, 8, 0, ymax])
    plt.grid(b=True, axis='y')
    plt.title("Nearest Neighbor", fontsize = 12)
    remove_border()
    
    plt.subplot(122)
    plt.bar(ind, bias2, color = red, label = 'Bias', width=width)
    plt.bar(ind, var2, bottom=bias2, color = blue, label='Var', width=width)
    plt.xticks(ind+.5, xlabel, rotation='horizontal')
    plt.xlabel('K')
    plt.ylabel('')
    plt.legend(loc='upper right')
    plt.axis([0, 8, 0, ymax])
    plt.grid(b=True, axis='y')
    plt.title("Prototype based NN", fontsize=12)
    plt.suptitle(prob, fontsize=14)
    remove_border()
    plt.show()
    
def test_stacked_bar_graph():
    bias1 = [0.040, 0.035, 0.039, 0.031, 0.041, 0.031, 0.029, 0.030]
    var1 = [0.004, 0.005, 0.005, 0.005, 0.000, 0.002, 0.002, 0.001]
    bias2 = [0.045, 0.038, 0.038, 0.027, 0.027, 0.030, 0.035, 0.033]
    var2 = [0.015, 0.005, 0.028, 0.007, 0.003, 0.000, 0.003, 0.003] 
    stacked_bar_graph(15, bias1, var1, bias2, var2, prob="Breast Cancer");
    
class PlotInstances:
   '''
   Class PlotInstance plots class data with
          - using the center
          - using the selected prototypes

   Parameters
   ----------
   X: R X C array (default R=10, C=2)
   Y: no. of classes(K=3)
   '''
   def __init__(self, X, Y):
       self.X = X; self.Y = Y;
       self.cls = np.unique(self.Y);
       self.K = len(self.cls);
       self.N = len(self.Y)/len(self.cls);

       # set the axis
       xgap = np.ptp(self.X[:,0])/50.0; ygap = np.ptp(self.X[:,1])/50.0; 
       xmin = self.X[:,0].min(); xmax = self.X[:,0].max();
       ymin = self.X[:,1].min(); ymax = self.X[:,1].max();
       if np.abs(xmax-xmin) > np.abs(ymax-xmin):
          self.axis = [ xmin-xgap, xmax+xgap, xmin-ygap, xmax+ygap];
       else:
          self.axis = [ ymin-xgap, ymax+xgap, ymin-ygap, ymax+ygap];
       #print("axis = ", self.axis);
       plt.rc('font', family='serif');
       plt.rc('xtick', labelsize='x-small');
       plt.rc('ytick', labelsize='x-small');

   def plotWithRawData(self,subfig='111'):
       ' contour X and Y as it is'
       marker = ["ro", "k+", "bx", "ko", "b+", "rx"];
       plt.subplot(subfig);
       for ea in range(self.K):
           # be caution! It returns a tuple with n-dimensions.
           idx = np.where(self.Y==ea)[0];
           plt.plot( self.X[idx,0], self.X[idx,1], marker[ea] );
       plt.axis(self.axis);

   def plotWithCenters(self,subfig='111'):
       ' contour X and Y with class centers '
       marker = ["ro", "k+", "bx", "ko", "b+", "rx"];
       plt.subplot(subfig);
       for ea in range(self.K):
           # be caution! It returns a tuple with n-dimensions.
           idx = np.where(self.Y==ea)[0];
           cen = np.mean(self.X[idx], axis=0);
           #print 'cen = ', cen
           plt.plot( self.X[idx,0], self.X[idx,1], marker[ea] );
           plt.plot( cen[0], cen[1], "kD", markersize=5);
           nc = plt.Circle(cen, 1, color='b', clip_on=True, fill=False);
           fig = plt.gcf();
           fig.gca().add_artist(nc);
       plt.axis(self.axis);

   def plotWithRadius(self, L, ep, subfig="111"):
       ' contour X and Y with len(L) prototypes and radius vector Ep '
       marker = ["ro", "k+", "bx", "ko", "b+", "rx"];
       plt.subplot(subfig);
       for ea in np.arange(self.K):
           # be caution! It returns a tuple with n-dimensions.
           idx = np.where(self.Y==ea)[0];
           plt.plot( self.X[idx,0], self.X[idx,1], marker[ea]);
           sidx = np.intersect1d(idx, L);
           plt.plot( self.X[sidx,0], self.X[sidx,1], marker[ea], markersize=10, markeredgecolor='black', markeredgewidth=2 );

           #for i in idx:
           #    plt.text(self.X[i,0]+.03, self.X[i,1]+.03,i)

       fig = plt.gcf();
       for ea in L:
           nc = plt.Circle(self.X[ea], ep, color ='b', clip_on=True, fill=False);
           fig.gca().add_artist(nc);
       plt.axis(self.axis);
       plt.title('Drawing graph with radius %5.2f ' % ep)
       #print len(L), " circles"

   def plotWithRadii(self, L, Ep, subfig="111"):
       ' contour X and Y with len(L) prototypes and their seperated radii Ep '
       marker = ["ro", "k+", "bx", "ko", "b+", "rx"];
       plt.subplot(subfig);
       for ea in np.arange(self.K):
           # be caution! It returns a tuple with n-dimensions.
           idx = np.where(self.Y==ea)[0];
           plt.plot( self.X[idx,0], self.X[idx,1], marker[ea] );
           sidx = np.intersect1d(idx, L);
           plt.plot( self.X[sidx,0], self.X[sidx,1], marker[ea], markersize=10, markeredgecolor='black', markeredgewidth=2 );

           # display data indices
           #for i in idx:
               #plt.text(self.X[i,0]+.03, self.X[i,1]+.03,i)

       fig = plt.gcf();
       for ea in L:
           nc = plt.Circle(self.X[ea], Ep[ea], color ='b', clip_on=True, fill=False);
           fig.gca().add_artist(nc);
       #plt.title('Drawing graph with variable radii');
       plt.axis(self.axis);
       #print len(L), " circles"

   def plotWithRectangles(self, Hr=None, subfig="111"):
       ' contour X and Y with selected rectangles Hr '

       assert Hr is not None, "Require the set of rectangles"

       marker = ["ro", "k+", "bx", "ko", "b+", "rx"];
       plt.subplot(subfig);
       for ea in np.arange(self.K):
           # be caution! It returns a tuple with n-dimensions.
           idx = np.where(self.Y==ea)[0];
           plt.plot( self.X[idx,0], self.X[idx,1], marker[ea] );
           #sidx = np.intersect1d(idx, L);
           plt.plot( self.X[idx,0], self.X[idx,1], marker[ea], markersize=1, markeredgecolor='black', markeredgewidth=0 );

           # display data indices
           #for i in idx:
               #plt.text(self.X[i,0]+.03, self.X[i,1]+.03,i)

       fig = plt.gcf();
       cnt = 0
       for hr in Hr:
           cnt = cnt + 1
           idx = hr.getY()
           (width,  height) = hr.xmax - hr.xmin
           nc = plt.Rectangle(hr.xmin, width, height, color ='b', clip_on=True, fill=False);
           # draw new data points
           plt.plot( hr.xmid[0], hr.xmid[1], 'x', markersize=5, color=marker[idx][0], markeredgewidth=2)
           fig.gca().add_artist(nc);
       plt.title('Prototype selection based on hyper-rectangles');
       plt.axis(self.axis);
       print(cnt, " rectangles <<<<< ")

   def draw_graph(self):
       plt.show();

   def clear_graph(self):
       plt.cls();

   def findItem(self, ith):
       ' return the length of same class of nearest neighbors '
       try:
           assert(0 <= ith<self.N)
       except IndexError as e:
           print('Error: ', e)
           sys.exit(1); # abort

       L = list();
       for ea in np.arange(self.K):
           idx = np.where(self.Y==ea)[ith];# the first index
           L.append(idx[0]);
       return L;

def plot_by_rect():
   assert len(sys.argv) >= 2

   dd = ReadInstances(filename = sys.argv[1]).getDataObj()

   X, y = dd.getXy();

   dp = PlotInstances( X, y );
   dp.plotWithRectangles(range(X.shape[0]));
   dp.draw_graph();
  
def plot_by_fixed():
   assert len(sys.argv) >= 2

   dd = ReadInstances(filename = sys.argv[1]).getDataObj()

   X, y = dd.getXy();

   dp = PlotInstances( X, y );
   dp.plotWithRadius(range(X.shape[0]), .5);
   dp.draw_graph();
  
def plot_in_raw():
   assert len(sys.argv) >= 2

   dd = ReadInstances(filename = sys.argv[1]).getDataObj();

   X, y = dd.getXy();

   dp = PlotInstances( X, y );
   dp.plotWithRawData();
   dp.draw_graph();
  
if __name__ == "__main__":
   plot_in_raw()
#   test_stacked_bar_graph()
#   L = np.random.choice(np.arange(Y.size), size = 20, replace=False);
#   dp.plotWithRadius(L, 0.1);
#   dp.draw_graph();

#   ep = np.zeros( Y.size, dtype=float);
#   ep[L] = np.random.ranf()+1;
#   print 'L.size= ', L.size, ' ep.size = ', ep.size
#   dp.plotWithRadii(L, ep);
#   dp.draw_graph();
