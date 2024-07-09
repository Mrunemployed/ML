import numpy as np
import random
import time


class multiLR():

    def __init__(self,iter,**kwargs) -> None:
        if len(kwargs.keys()) >0 and "x_train" in kwargs.keys() and "y_train" in kwargs.keys():
            self.x_train = kwargs['x_train']
            self.y_train = kwargs['y_train']
            self.iter = iter
        else:
            print("No training sets given")


    def derivative(self,model,w,b):
        x_t = self.x_train
        y_t = self.y_train
        m = self.x_train.shape[0]
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
            return sum_w,sum_b
        

    def learn(self,model):
    # Implementing gradient descent
        alpha = 1.0e-2
        if model == "multiple":
            pass
        else:
            dq,w,db,b = 0.,0.,0.,0.
            for i in range(self.iter):
                dw,db = self.derivative("single",w,b)
                w = w - alpha*dw
                b = b - alpha*db
                if i%1000 == 0 :
                    print(f"w : {w} , b : {b}")
            return w,b


    def multiple_LR():
        pass
    
    

    def univar_LR(self,x):
        w,b = self.learn('single')
        model = w*x + b
        return model


    def main(self,x):
        if len(self.x_train.shape) == 2:
            w_v,b = self.multiple_LR(x)
        else:
            res = self.univar_LR(x)
            print(f"Prediction : ${res:.2f} thousand")
            return res

x_train = np.array([1.0,2.0,3.0,4.0,1.5,0.9])
y_train = np.array([300.0,500.0,600.0,800.0,400.0,200.0])
pred = multiLR(10000,x_train=x_train,y_train=y_train)
pred.main(1.3)