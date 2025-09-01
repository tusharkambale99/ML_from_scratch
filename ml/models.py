import numpy as np

def L1_distance(x1,x2):
    return np.sum(abs(x1-x2),axis=1)

def L2_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2,axis=1))

def sigmoid(x):
    return 1/(1+np.exp(-x))

class KNearestNeighbors:
    def __init__(self):
        pass
    
    #X is of the shape (m(examples),n(dimension))
    #Y is of the shape(m,1)
    def train(self,X,Y,k,distance):
        self.X_train = X
        self.Y_train = Y
        self.k = k
        self.distance = distance
        
    def predict(self,X):
        y = []
        for x in X:
            x = x.reshape(1,x.shape[0])
            d = self.distance(self.X_train, x)
            idx = np.argsort(d)[:self.k]
            knn = np.argmax(np.bincount(self.Y_train[idx].squeeze()))
            y.append([knn])
        return np.array(y)

    def test(self,X,Y):
        y = self.predict(X)
        accuracy = np.sum(((Y-y) == 0))/y.shape[0]
        return accuracy   
    

class LinearRegression:
    def __init__(self):
        pass
    
    #X is of the shape (m(examples),n(dimension))
    #Y is of the shape(m,1)
    def train(self,X,Y,learning_rate=0.1,number_of_iterations=100):
        self.n = X.shape[1]
        self.W = np.zeros([self.n,1])
        self.b = 0
        
        for x in range(0,number_of_iterations):
            A = np.dot(X,self.W) + self.b
            #SE_loss = (A-Y)**2
            dW = np.sum(2*(A-Y)*X,axis=0,keepdims=True).T
            db = np.sum(2*(A-Y))
            self.W -= learning_rate*dW
            self.b -= learning_rate*db
        
    def predict(self,X):
        A = np.dot(X,self.W) + self.b
        return A

    def test(self,X,Y):
        A = self.predict(X)
        MSE = np.sqrt(np.sum((A-Y)**2))
        return MSE

    
class LogisticRegression:
    def __init__(self):
        pass
    
    #X is of the shape (m(examples),n(dimension))
    #Y is of the shape(m,1)
    def train(self,X,Y,learning_rate=0.1,number_of_iterations=100):
        self.n = X.shape[1]
        self.W = np.zeros([self.n,1])
        self.b = 0
        
        for x in range(0,number_of_iterations):
            Z = np.dot(X,self.W) + self.b
            A = sigmoid(Z)
            #BCE_loss = -(Y*np.log(A)+(1-Y)*np.log(1-A))
            #dZ = A-Y
            dW = np.sum((A-Y)*X,axis=0,keepdims=True).T
            db = np.sum((A-Y))
            self.W -= learning_rate*dW
            self.b -= learning_rate*db
        
    def predict(self,X):
        Z = np.dot(X,self.W) + self.b
        A = sigmoid(Z)
        return A

    def test(self,X,Y):
        A = self.predict(X)
        pred = np.round(A)
        accuracy = np.sum(((Y-pred) == 0))/Y.shape[0]
        return accuracy  
    
        