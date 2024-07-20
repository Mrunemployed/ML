import numpy as np
import copy

class predictM():

    def __init__(self,x_t,y_t,itr:int) -> None:
        self.x_t = x_t
        self.y_t = y_t
        self.itr = itr

    def gradient_derivative(self,w,b):
        m,n = self.x_t.shape
        x_t = self.x_t
        y_t = self.y_t
        dw = copy.deepcopy(w)
        db = b
        for i in range(m):
            j_wb = np.dot(w,x_t[i]) +b - y_t[i]
            for j in range(n):
                dw[j] = dw[j] + j_wb * x_t[i][j]
            db = db+j_wb
        return dw/m,db/m
    
    def gradient_descent(self):
        m,n = self.x_t.shape
        alpha = 5.0e-7
        w = np.zeros(n)
        b = 0.
        
        for r in range(self.itr):
            tw,tb = self.gradient_derivative(w,b)
            w = w - alpha*tw
            b = b - alpha*tb
        return w,b
    
    def model(self,x_pred,w,b):
        pdt = np.dot(w,x_pred) + b
        return pdt

    def predict(self,x_in):
        w,b = self.gradient_descent()
        res = self.model(x_in,w,b)
        return res
    

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
c = predictM(X_train,y_train,1)
res = c.predict(X_train[1])
print(f"${res:.4f} thousand")

n = 6
div = int(n/4)
rem = n%4
print(div*(20) + rem*4)
print(4*n + 4*div)