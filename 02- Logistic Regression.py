import numpy as np

class LogisticRegression():
    
    def __init__(self,lr,iterations):
        
        self.lr = lr
        self.iterations = iterations
        
    def sigmoid(self,x):
        
        return 1 / (1 + np.exp(-x))
        
    def fit(self,X,Y):
        
        samples_num , features_num = X.shape
        
        self.weights = np.random.rand(features_num)
        self.bais = 1
        
        for i in range(self.iterations):
            y_pred = np.dot(X,self.weights) + self.bais
            y_pred = self.sigmoid(y_pred)
            
            error = y_pred - Y
            dW =   np.sum(2*X.T.dot(error)) / samples_num
            db = np.sum(error) / samples_num
            
            self.weights -=  dW * self.lr
            self.bais -=  db * self.lr
    
    def predict(self,X):
        net = np.dot(X,self.weights) + self.bais
        net = self.sigmoid(net)
        net = np.round(net)
        return net
    
    def get_accuracy(self,y_true,y_pred):
        correct= 0
        total = len(y_true)
        
        for i in range(total):
            if y_true[i] == y_pred[i]:
                correct +=1
        
        
        accuracy = correct / total
        
        return accuracy