import numpy as np
import copy

class Predict_linear():

    def __init__(self,x_train,y_train,iterations) -> None:
        self.x_t = x_train
        self.y_t = y_train
        self.iterations = iterations
        if len(self.x_t.shape) == 2:
            self.model = 'multiple'
            self.w = np.zeros(self.x_t.shape[1])
        else:
            self.w = 0.
            self.model = 'single'
        self.b = 0.

    def compute_gradient(self,w,b):
        x_t = copy.deepcopy(self.x_t)
        y_t = copy.deepcopy(self.y_t)

        if self.model == 'multiple':
            m,n = x_t.shape
            dw = np.zeros((n,))
            db = 0.
            for i in range(m):
                j_wb = np.dot(w,x_t[i]) + b - y_t[i]
                db = db + j_wb
                for j in range(n):
                    dw[j] = dw[j] + j_wb*x_t[i,j]
            return dw/m,db/m
        else:
            m = x_t.shape[0]
            dw,db = 0.,0.
            for i in range(m):
                j_wb = (w*x_t[i] + b) - y_t[i]
                db = db + j_wb
                dw = dw + j_wb*x_t[i]
            return dw/m,db/m
    
    def learn_gradient_descent(self,w,b,alpha):
        # the function implementing gradient descent
        for i in range(self.iterations):
            dw,db = self.compute_gradient(w,b)
            w = w - alpha*dw
            b = b - alpha*db
        print("learn_gradient: ",w,b)
        return w,b
    
    def _model(self,w,b,x_in):
        if self.model == 'single':
            p = w*x_in + b
            return p
        else:
            p = np.dot(w,x_in) + b
            return p

    def predict(self,x_in):
        print("model: ",self.model)
        if self.model == 'single':
            # for univariate linear regression
            w = self.w
            b = self.b
            alpha = 1.0e-2
            w,b = self.learn_gradient_descent(w,b,alpha)
            result = self._model(w,b,x_in)
            return result
        else:
            # for multivalue linear regression
            w = self.w
            b = self.b
            alpha = 5.0e-7
            w,b = self.learn_gradient_descent(w,b,alpha)
            result = self._model(w,b,x_in)
            return result

x_train = np.array([1.0,2.0,3.0,4.0,1.5,0.9])
y_train = np.array([300.0,500.0,600.0,800.0,400.0,200.0])
pred = Predict_linear(x_train,y_train,1000)
prediction = pred.predict(1.3)
print(f"prediction: {prediction:.2f}")

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
multi_predict = Predict_linear(X_train,y_train,10000)
prediction = multi_predict.predict(X_train[1])
print(f"prediction: {prediction:.4f}")
