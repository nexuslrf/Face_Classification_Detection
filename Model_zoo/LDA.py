import numpy as np
import matplotlib.pyplot as plt
import tqdm

class ClassData():
    def __init__(self, X, label, mean_all):
        self.X = X
        self.label = label
        self.num = X.shape[1]
        self.mean = X.mean(1)
        self.var = np.cov(X)
        self.Swi = self.var * (self.num - 1)# unbiased estimation    
        tmp = self.mean - mean_all
        self.cov = np.dot(tmp, tmp.transpose())
        self.Sbi = self.cov * self.num

class LDA():
    def __init__(self, dim, num_classes=2, labels=[-1,1], bayes = True):
        self.dim = dim+1
        self.num_classes = num_classes
        self.labels = range(num_classes) if labels is None else labels
        self.bayes = bayes

    
    def fit(self, X,y):
        if X.shape[1]!=len(y):
            X = X.transpose()  # dim, num
        # add intercept
        X = np.insert(X, 0, values=1, axis=0)
        if self.bayes:
            self.mean_all = X.mean(1)
        else:
            self.mean_all = np.zeros(self.dim)
            
        self.Sw = np.zeros([self.dim, self.dim])
        diff = np.zeros(self.dim)
        for i in self.labels:
            X_c = ClassData(X[:,y==i], i, self.mean_all)
            self.Sw = self.Sw + X_c.Swi
            diff = diff + i * X_c.mean
            if not self.bayes:
                self.mean_all + X_c.mean
        
        self.Sb = np.dot(diff, diff.transpose())
        
        if not self.bayes:
            self.mean_all /= self.num_classes
        # For Binary Case    
        self.w = np.dot(np.linalg.pinv(self.Sw),diff.reshape(self.dim,1))
        # Standardize
#         scale = np.dot(self.w.transpose(), self.Sw, self.w)
#         self.w /= np.sqrt(scale)
        self.mean_w = np.dot(self.w.transpose(), self.mean_all)
    
    def predict(self, X):
        if X.shape[0]!=self.dim and X.shape[0]!=self.dim-1:
            X = X.transpose()
        if X.shape[0]==self.dim-1:
            X = np.insert(X, 0, values=1, axis=0)  
        pred = np.dot(self.w.transpose(),X) - self.mean_w
        pred[pred>=0] = 1
        pred[pred<0] = -1
        return pred.astype(int)
    
    def get_acc(self, X, y):
        return (self.predict(X) == y).sum()/len(y)
    
    def get_score(self, X):
        if X.shape[0]!=self.dim and X.shape[0]!=self.dim-1:
            X = X.transpose()
        if X.shape[0]==self.dim-1:
            X = np.insert(X, 0, values=1, axis=0)  
        pred = np.dot(self.w.transpose(),X) - self.mean_w
        return pred
        
    def get_variance(self):
        intra_var = self.w.transpose().dot(self.Sb).dot(self.w)
        inter_var = self.w.transpose().dot(self.Sw).dot(self.w)
        return intra_var, inter_var
            