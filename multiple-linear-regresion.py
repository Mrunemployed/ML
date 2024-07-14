import numpy as np
import random
import time
import copy

class multiLR():

    def __init__(self,iter,**kwargs) -> None:
        if len(kwargs.keys()) >0 and "x_train" in kwargs.keys() and "y_train" in kwargs.keys():
            self.x_train = kwargs['x_train']
            self.y_train = kwargs['y_train']
            self.iter = iter
        else:
            print("No training sets given")
            

    def derivative(self,model,w,b,m,**kwargs):
        x_t = self.x_train
        y_t = self.y_train
        sum_w,sum_b = 0.,0.
        if model == "multiple":
            pass
        else:
            sum_w,sum_b = 0.0,0.0
            for i in range(m):    
                f_wb_x = (w*x_t[i] + b) - y_t[i]
                sum_w = sum_w + f_wb_x*x_t[i]
                sum_b = sum_b + f_wb_x
            sum_w = sum_w/m
            sum_b = sum_b/m
            print(f"sum_w: {sum_w}, sum_b: {sum_b}")
            return sum_w,sum_b
        

    def learn(self,model):
    # Implementing gradient descent
        x_t = self.x_train
        y_t = self.y_train
        m = x_t.shape[0]
        if model == "multiple":
            alpha = 5.0e-7
            m,n = x_t.shape
            w = np.zeros((n,))
            b = 0.
            for r in range(self.iter):
                dw = np.zeros((n))
                db = 0.
                for i in range(m):
                    j_wb = (np.dot(x_t[i],w) + b) - y_t[i]
                    for j in range(n):
                        dw[j] = dw[j] + j_wb * x_t[i,j]
                    db = db + j_wb
                dw = dw/m
                db = db/m
                w = w - alpha*dw
                b = b - alpha*db

            return w,b
            
        else:
            dw,w,db,b = 0.,0.,0.,0.
            alpha = 1.0e-2
            for i in range(self.iter):
                dw,db = self.derivative("single",w,b,m)
                w = w - alpha*dw
                b = b - alpha*db
                if i%1000 == 0 :
                    print(f"w : {w} , b : {b}")
            return w,b


    def multiple_LR(self,x:np.ndarray):
        w_in = np.zeros((4,))
        b_in = 0.
        w,b = self.learn('multiple')
        print("+"*80)
        print(f"final w: {w}, b: {b}")
        model = np.dot(w,x) + b
        return model
    
    def univar_LR(self,x):
        w,b = self.learn('single')
        model = w*x + b
        return model


    def main(self,x):
        if len(self.x_train.shape) == 2:
            res = self.multiple_LR(x)
            print("Multiple features prediction....")
        else:
            res = self.univar_LR(x)
            print("Univariate")
        print(f"Prediction : ${res} thousand")
        return res

# x_train = np.array([1.0,2.0,3.0,4.0,1.5,0.9])
# y_train = np.array([300.0,500.0,600.0,800.0,400.0,200.0])
# pred = multiLR(1000,x_train=x_train,y_train=y_train)
# pred.main(1.4)
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
pred = multiLR(1000,x_train=X_train,y_train=y_train)
pred.main(X_train[1])