import numpy as np
import matplotlib.pyplot as plt
import math
# plt.style.use('./deeplearning.mpstyle')

# X_train -> the features or inputs for training size in in 1000 sqft.
# y_train -> the outputs or the target for training, price in 1000s of dollars.
x_train = np.array([1.0,2.0])
y_train = np.array([300.0,500.0])

# prints the shape of the array, The shape of a NumPy array is a tuple that indicates the size of the array along each dimension. 
# In this case, a is a 1-dimensional array with 2 elements, so its shape is (2,).
print("shape: ",x_train.shape)
# print(x_train[1],y_train[1])

# x_train is the x array or the x features, y_train is the target or output, marker-> denotes the type of marker,
# like x,o,tick etc. c-> color here 'r' denotes red
# In this context x_train is the x-axis and y_train is the y-axis - builds a scatter chart
plt.scatter(x_train,y_train,marker='x',c='r')

plt.title("scatter this")

plt.ylabel("Price in 1000s")

plt.xlabel("Size in 1000 SQFT")
# plt.show()

w = 200
b = 100

def compute_model_output(w,x,b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        # This is our model through which we have established a relationship with our x_train and y_train
        f_wb[i] = w*x[i] + b

    return f_wb
# We have tweaked the value of w so that the prediction fits our output data
tmp_f_wb = compute_model_output(w,x_train,b)
print(tmp_f_wb)

plt.plot(x_train,tmp_f_wb,c='b',label='Our Prediction')
plt.scatter(x_train,y_train,marker='x',c='r',label="Actual Values")
plt.legend()
# plt.show()

# so, utilizing this model
# x_i is the new feature for which we are trying to predict the actual price.
x_i = 2.3
calculate_price = w*x_i +b
print(f"{calculate_price:.0f} thousand dollars")


# Here starts the real deal!

x_train = np.array([1.0,2.0])
y_train = np.array([300.0,500.0])

class UnilinearDescent():
    
    def __init__(self,iterations:int):
        self.iterations = iterations
        self.w = 0.0
        self.b = 0.0
        self.x_train = np.array([1.0,2.0,3.0,4.0,1.5,0.9])
        self.y_train = np.array([300.0,500.0,600.0,800.0,400.0,200.0])


    def cost_fn(self):
        pass

    def derivative_fn(self):
        m = self.x_train.shape[0]
        x_t = self.x_train
        y_t = self.y_train
        w = self.w
        b = self.b
        sum_w = 0.0
        sum_b = 0.0
        for i in range(m):
            f_wbx = w*x_t[i] + b
            sum_w = sum_w + (f_wbx - y_t[i]) * x_t[i]
            sum_b = sum_b + (f_wbx - y_t[i])
        # print("sum_w: ",sum_w/m,"\nsum_b: ",sum_b/m)
        return sum_w/m,sum_b/m
        
    def grad_desc(self,alpha:float,iter:int):
        # w = 1/m sum((f_wb(xi)-yi)xi
        # b = 1/m sum(f_wb(xi) - yi)
        tmp_w = self.w
        tmp_b = self.b
        m = x_train.shape[0]
        for i in range(iter):
            tmp_w,tmp_b = self.derivative_fn()
            self.w = self.w - alpha*tmp_w
            self.b = self.b - alpha*tmp_b
            if i% math.ceil(iter/10) == 0:
                print(f"w: {self.w: 0.3e}, b:{self.b: 0.5e}")
        print(f"w:{self.w:.4f} b:{self.b:.4f}")
    
    def predict_cost(self,x_in):
        tmp_alpha = 1.0e-2
        self.grad_desc(alpha=tmp_alpha,iter=self.iterations)
        cost = self.w*x_in + self.b
        return cost

b = UnilinearDescent(10000)
prediction = b.predict_cost(1.3)
print(f"prediction: ${prediction:0.1f} thousand")

