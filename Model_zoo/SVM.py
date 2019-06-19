from sklearn.svm import SVC

class SVM():
    def __init__(self, params={}):
        self.clf = SVC(
            kernel = self.set_params('rbf', 'kernel', params),
            gamma = self.set_params('scale', 'gamma', params),
            verbose = self.set_params(False, 'verbose', params)
                      )
        
    def set_params(self, default, label, params):
        return default if label not in params.keys() else params[label]

    
    def fit(self, X, y):
        if X.shape[1]==len(y):
            X = X.transpose()  # num, dim
        self.dim = X.shape[1]
        self.clf.fit(X,y)
        
    def get_acc(self, X, y):
        if X.shape[1]==len(y):
            X = X.transpose()  # num, dim
        return self.clf.score(X,y)
    
    def predict(self, X):
        if X.shape[1] != self.dim:
            X = X.transpose()
        return self.clf.predict(X)
    
    def get_score(self, X):
        if X.shape[1] != self.dim:
            X = X.transpose()
        return self.clf.decision_function(X)
    
    def get_SV(self):
        return self.clf.support_