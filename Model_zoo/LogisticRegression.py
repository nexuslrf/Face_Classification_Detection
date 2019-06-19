import numpy as np
import matplotlib.pyplot as plt
import tqdm

class logisticRegression():
    def __init__(self, dim, alpha=1.0, zero_init=False):
        self.dim = dim+1
        self.alpha = alpha
        self.theta = np.zeros([dim+1]) if zero_init else np.random.randn(dim+1)
#         self.theta[0] = 0
        self.sigmoid = lambda x: 1/(1+np.exp(-x))
        self.h = lambda x: self.sigmoid(np.dot(self.theta,x))
    
    def fit(self, X, y, params=dict()):
        if X.shape[1]!=len(y):
            X = X.transpose()
        # add intercept
        X = np.insert(X, 0, values=1, axis=0)
        yX = y*X
        self.optim = self.set_params('SGD', 'optimizer', params)
        if self.optim == 'SGD':
            self.SGD(X,y,yX,params)
        elif self.optim == 'GD':
            self.GD(X,y,yX,params)    
        elif self.optim == 'SLGD':
            self.SLGD(X,y,yX,params)
        else:
            raise NameError(self.optim)
        
        
    def get_acc(self, X, y):
        return ((self.predict(X) * y)>0).sum()/len(y)
    
    def set_params(self, default, label, params):
        return default if label not in params.keys() else params[label]
        
    def GD(self,X, y, yX, params):
        rounds = self.set_params(50, 'rounds', params)
        decay = self.set_params(1.0, 'decay', params)
        L2 = self.set_params(0.5, 'l2_ratio', params)
        
        rnd = tqdm.trange(rounds, desc='Logisitic GD')            
        for i in rnd:
            dloss =  np.dot((self.h(yX)-1),yX.transpose())
            dloss if L2 == 0 else dloss + L2 * self.theta
            self.theta = self.theta - self.alpha * dloss
            self.acc = self.get_acc(X,y)
            rnd.set_description('Logistic GD (acc={:.4f})'.format(self.acc))
            self.alpha *= decay
        
    
    def SGD(self, X, y, yX, params):
        rounds = self.set_params(50, 'rounds', params)
        batchsize = self.set_params(32, 'batchsize', params)
        decay = self.set_params(1.0, 'decay', params)
        L2 = self.set_params(0.5, 'l2_ratio', params)
        
        num_item = yX.shape[1]
        num_batch = num_item//batchsize
        left_batch = num_item%batchsize
        num_batch = num_batch if left_batch == 0 else num_batch+1
        rnd = tqdm.trange(rounds, desc='Logisitic SGD')
        for i in rnd:
            shuffle = np.random.permutation(np.arange(num_item))
            for b in range(num_batch):
                if b!=num_batch-1:
                    batch = yX[:,shuffle[b*batchsize:(b+1)*batchsize]]
                else:
                    batch = yX[:,shuffle[b*batchsize:]]
                dloss = np.dot((self.h(batch)-1),batch.transpose())
                dloss = dloss if L2 == 0 else dloss + L2 * self.theta
                self.theta = self.theta - self.alpha * dloss
                
            self.acc = self.get_acc(X,y)
            rnd.set_description('Logistic SGD (acc={:.4f})'.format(self.acc))
            self.alpha *= decay    
    
    def SLGD(self, X, y, yX, params):
        rounds = self.set_params(50, 'rounds', params)
        batchsize = self.set_params(32, 'batchsize', params)
        decay = self.set_params(0.99, 'decay', params)
        
        num_item = yX.shape[1]
        num_batch = num_item//batchsize
        left_batch = num_item%batchsize
        num_batch = num_batch if left_batch == 0 else num_batch+1
        rnd = tqdm.trange(rounds, desc='Logisitic SLGD')
        for i in rnd:
            shuffle = np.random.permutation(np.arange(num_item))
            for b in range(num_batch):
                if b!=num_batch-1:
                    batch = yX[:,shuffle[b*batchsize:(b+1)*batchsize]]
                else:
                    batch = yX[:,shuffle[b*batchsize:]]
                dloss = np.dot((self.h(batch)-1),batch.transpose())
                # variance for Langevin Noise term is sqrt(lr)
                dloss = dloss + np.sqrt(self.alpha) * np.random.randn(self.dim)
                self.theta = self.theta - self.alpha * dloss
                
            self.acc = self.get_acc(X,y)
            rnd.set_description('Logistic SLGD (acc={:.4f})'.format(self.acc))
            self.alpha *= decay    
    
    
    def predict(self, X):
        if X.shape[0]!=self.dim and X.shape[0]!=self.dim-1:
            X = X.transpose()
        if X.shape[0]==self.dim-1:
            X = np.insert(X, 0, values=1, axis=0)            
        pred = self.h(X)
        pred[pred<0.5] = -1
        pred[pred>=0.5] = 1
        return pred.astype(int)

    def get_score(self, X):
        if X.shape[0]!=self.dim and X.shape[0]!=self.dim-1:
            X = X.transpose()
        if X.shape[0]==self.dim-1:
            X = np.insert(X, 0, values=1, axis=0)            
        pred = self.h(X)
        return pred