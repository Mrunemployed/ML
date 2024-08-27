import numpy as np
import matplotlib.pyplot as plt
import copy

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
# X_train = np.array([[3204, 4, 1, 10],[2104, 5, 1, 45], [1416, 3, 2, 40],[1104, 2, 1, 5], [852, 2, 1, 35]])
# y_train = np.array([521, 460, 232, 300, 178 ])

class fscale():

    def __init__(self,x_t,y_t) -> None:
        self.x_t = x_t
        self.y_t = y_t
        self.mu = np.zeros((x_t.shape[1]))
        self.r = np.zeros((x_t.shape[1]))

    def set_mean_range(self):
        x_t = self.x_t
        m,n = x_t.shape
        mu = np.mean(x_t,axis=0)
        max_a = np.max(x_t,axis=0)
        min = np.min(x_t,axis=0)
        self.mu = mu
        self.r = max_a-min

    def mean_scale(self,**kwargs):
        if kwargs and "x_t" in kwargs.keys():
            x_t = kwargs['x_t']
        else:
            x_t = self.x_t
        scaled_x_t = (x_t - self.mu)/self.r
        # Display the normalized vs original as a hist graph
        # self.scatterplot(x_t,scaled_x_t,"Original vs scaled")
        # self.norm_plot(original=x_t,scaled=scaled_x_t)
        print(x_t,'\n','-'*50,'\n',scaled_x_t)
        return scaled_x_t
        

    def zscore_normalize_features(self):
        """ computes  X, zcore normalized by column Args:
        X (ndarray (m,n))     : input data, m examples, n features
        Returns:
        X_norm (ndarray (m,n)): input normalized by column
        mu (ndarray (n,))     : mean of each feature
        sigma (ndarray (n,))  : standard deviation of each feature
        """
        # find the mean of each column/feature
        mu     = np.mean(self.x_t, axis=0)                 # mu will have shape (n,)
        # find the standard deviation of each column/feature
        sigma  = np.std(self.x_t, axis=0)                  # sigma will have shape (n,)
        # element-wise, subtract mu for that column from each example, divide by std for that column
        X_norm = (self.x_t - mu) / sigma      

        return (X_norm, mu, sigma)
 
    

    def scatterplot(self,x_t,scaled_x_t, title):
        fig,ax = plt.subplots(1, 2, figsize=(12, 3))
        ax[0].scatter(x_t[:,0], x_t[:,3])
        ax[1].scatter(scaled_x_t[:,0], scaled_x_t[:,3])
        plt.title(title)
        plt.axis('equal')
        plt.show()
        """
        The plot above shows the relationship between two of the training set parameters, "age" and "size(sqft)". *These are plotted with equal scale*. 
        - Left: Unnormalized: The range of values or the variance of the 'size(sqft)' feature is much larger than that of age
        - Middle: The first step removes the mean or average value from each feature. This leaves features that are centered around zero. It's difficult to see the difference for the 'age' feature, but 'size(sqft)' is clearly around zero.
        - Right: The second step divides by the standard deviation. This leaves both features centered at zero with a similar scale.

        """

    def norm_plot(self, **data):
        if data:
            l = len(data.keys())
            fig,ax = plt.subplots(l, self.x_t.shape[1], figsize=(12,3))
            # print(fig,ax)
            for idx,(key,val) in enumerate(data.items()):
                d = data[key]
                for i in range(ax.shape[1]):
                    pass
                    print(d[:,i],"\n","-"*50)
                    ax[idx][i].hist(d[:,i], bins=10, color='blue', alpha=0.7)
                    ax[idx][i].set_xlabel(key)
                    ax[idx][i].hist(d[:,i], bins=10, color='blue', alpha=0.7)
                    ax[idx][i].set_xlabel(key)
                # ax[idx][0].set_ylabel("count")
            fig.suptitle("original vs scaled")
            plt.show()


# f = fscale(X_train,y_train)
# f.mean_scale()




class gradientDescent():

    def __init__(self,x_t,y_t,alpha,**kwargs):
        self.x_t = x_t
        self.y_t = y_t
        self.alpha = alpha
        if "itr" in kwargs.keys():
            self.itr = kwargs['itr']
        else:
            self.itr = 500

    def gradient_derivative(self,w,b,**kwargs):
        x_t = self.x_t
        y_t = self.y_t
        m,n = x_t.shape
        dw = np.zeros(n)
        db = 0.

        for i in range(m):
            error = np.dot(w,x_t[i]) + b -y_t[i]
            for j in range(n):
                dw[j] = dw[j]+(error*x_t[i][j])
            db = db+error
        return dw/m,db/m


    def cost_fn(self,w,b):
        x_t = self.x_t
        y_t = self.y_t
        m,n = x_t.shape
        su = 0.
        for i in range(m):
            error = np.dot(w,x_t[i])+b -y_t[i]
            su = su+error*error
        j_wb = su/(2*m)
        return j_wb

    def gradient_descent(self):
        x_t = self.x_t
        m,n = x_t.shape
        w = np.zeros(n)
        b = 0.
        hist = []
        for i in range(self.itr):
            dw,db = self.gradient_derivative(w,b)
            w = w - self.alpha*dw
            b = b - self.alpha*db
            if int(i%100) == 0 :
                print(f"w : {[x for x in w]}, b: {b:.4f}")
                cost = self.cost_fn(w,b)
                hist.append(cost)
        self.learning_curve(hist)
        return w,b
    
    def learning_curve(self,hist):
        iterations = [x for x in range(0, self.itr, 100)]
        plt.plot(iterations,hist,marker='o')
        plt.xlabel("iterations->")
        plt.ylabel('cost(j_wb)')
        plt.title("learning Curve")
        plt.grid(True)
        plt.show()
    
# Previous alpha taken -> 1e-7 - too large 1e-11 - too small 3e-9 - early convergence, 1e-9 just right
alpha = 1.537e-9
learn = gradientDescent(X_train,y_train,alpha=alpha,itr=1500)
w,b = learn.gradient_descent()


nalpha = 7e-3
scale = fscale(X_train,y_train)
scale.set_mean_range()
scaled_x_t = scale.mean_scale()
learn_norm = gradientDescent(scaled_x_t,y_train,alpha=nalpha,itr=600)
nw,nb = learn_norm.gradient_descent()
print(f"final w : {w}, b: {b:.4f}")
print(f"normalised final w : {nw},b: {nb:.4f}")

# Prediction analysis using a linear graph
resl = []
plt.scatter(X_train[:,0],y_train, marker='x', c='r', label='Actual Value' )
for i in range(X_train.shape[0]):
    res = np.dot(w,X_train[i])+b
    # When making the prediction the input set needs to be normalized as well
    res1 = np.dot(nw,scaled_x_t[i])+nb
    resl.append(res1)
    print(f"res: {res:.2f} actual: {y_train[i]}")
    print(f"res: {res1:.2f} actual: {y_train[i]}")

plt.legend()
plt.plot(X_train[:,0],resl,label='Prediction')
plt.show()

# Prediction using scaled x_in data
q = np.array([[100, 2, 1, 15]])
scale_q = scale.mean_scale(x_t=q)
print(f"rescaled : {scale_q}")
resq = np.dot(nw,scale_q[0])+nb
print(f"Predicted price: $ {resq:.2f}")

resunq = np.dot(w,q[0])+b
print(f"Predicted price: $ {resq:.2f}")