#This module contains activation functions, loss functions and neural network models written from scratch
import numpy as np

##loss functions
def binary_cross_entropy(A,y):
    L = -(y*np.log(A)+(1-y)*np.log(1-A))
    return L

def multiclass_cross_entropy(A,y):
    L = -np.sum(y*np.log(A),axis=0)
    return L

##activation functions (pth derivative of f(x))
def sigmoid(x,p):
    if p==0:
        return 1/(1+np.exp(-x))
    if p==1:
        return x*(1-x)

def tanh(x,p):
    if p==0:
        a = np.exp(x)
        b = np.exp(-x)
        return (a-b)/(a+b)
    if p==1:
        return 1-x**2
    
def relu(x,p):
    if p==0:
        return np.maximum(x,0)
    if p==1:
        return (x>0)*np.ones(x.shape)

def linear(x,p):
    if p==0:
        return x
    if p==1:
        return np.ones(x.shape)

def softmax(x,p):
    if p==0:
        return np.exp(x)/np.sum(np.exp(x),axis=0,keepdims=True)

##neural network models
class FeedForwardNN():
    #L[i] is the number of units in ith layer
    #g[i] is the activation for the ith layer
    def __init__(self,L,g):
        self.L = L
        self.g = g
        self.parameters = {}
        self.grads = {}
        for i in range(1,len(self.L)):
            self.parameters["W"+str(i)] = np.random.randn(self.L[i], self.L[i-1])*np.sqrt(2/(L[i-1]+L[i]))
            self.parameters["b"+str(i)] = 0
    
    def forward(self,X):
        #X is of the shape (n(dim),m)
        #Y is of the shape (c(classes),m)
        A = [X]
        for i in range(1,len(self.L)):
            Zi = np.dot(self.parameters["W"+str(i)],A[i-1]) + self.parameters["b"+str(i)]
            Ai = self.g[i](Zi,0)
            A.append(Ai)
        return A
    
    def backward(self,A,dZl):
        m = A[-1].shape[1]
        dZi_prev = dZl
        for i in reversed(range(1,len(self.L))):
            dZi = dZi_prev
            self.grads["dW"+str(i)] = (1/m)*np.dot(dZi,A[i-1].T)
            self.grads["db"+str(i)] = (1/m)*np.sum(dZi, axis=1, keepdims=True)
            dZi_prev = np.dot(self.parameters["W"+str(i)].T,dZi)*self.g[i-1](A[i-1],1)
    
    def train(self,X,Y,learning_rate,loss,num_of_iterations=1):
        m = Y.shape[1]
        costs = []
        for x in range(0,num_of_iterations):
            A = self.forward(X)
            try:
                L = loss(A[-1],Y)
                J = (1/m)*np.sum(L)
                costs.append(J)
            except Exception:
                print("Log loss error")
            #for log-loss
            dZl = A[-1]-Y
            #############
            self.backward(A,dZl)
            for i in range(1,len(self.L)):
                self.parameters["W"+str(i)] -= learning_rate*self.grads["dW"+str(i)]
                self.parameters["b"+str(i)] -= learning_rate*self.grads["db"+str(i)]
        return costs
    
    def predict(self,X):
        A = self.forward(X)
        return A[-1]