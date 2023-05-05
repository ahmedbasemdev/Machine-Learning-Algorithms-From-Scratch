import numpy as np


class LinearRegression():
    
    def __init__(self,lr,iterations):
        self.lr = lr
        self.iterations = iterations
        self.history = {"epoch":[],"loss":[]}

    def fit(self,x,y):
        n_samples , n_features = x.shape
        
        self.weights = np.random.randn(n_features)
        self.bais = 0
        
        for i in range(self.iterations):
            y_pred = np.dot(x,self.weights) + self.bais
            
            error = y_pred - y

            dW = (1 / n_samples) * np.dot(x.T,error)
            db = (1 / n_samples) * np.sum(error)

            self.weights = self.weights - self.lr * dW
            self.bais = self.bais - self.lr * db
            
            cost = self.score(x,y)
            self.history['epoch'].append(i+1)
            self.history['loss'].append(cost)
            
    def predict(self,x):
        
        net = np.dot(x,self.weights) + self.bais
        return net
     
    def score(self,x,y_true):
        y_pred = self.predict(x)
        
        error = (y_pred - y_true)**2
        mse = np.mean(error)
        rmse = np.sqrt(mse)

        return rmse
    
    def r2_score(self, x, y_true):
        
        y_pred = self.predict(x)
        sum_squared_regression = np.sum((y_true - y_pred)**2)
        total_sum_squares = np.sum((y_true - np.mean(y_true)) ** 2)
        
        r2 = 1 - (sum_squared_regression / total_sum_squares)
        return r2