import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)
# plt.style.use('./deeplearning.mplstyle')
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
scaler = StandardScaler()
x_t_scaled = scaler.fit_transform(X_train)

x_test = np.array([[-1.08413206, -0.44444444, -0.33333333, -2.5]])

print("before:",X_train)
print("Normalized:",x_t_scaled)

sgdr = SGDRegressor(max_iter=10000)
sgdr.fit(x_t_scaled,y_train)
print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")

# coef_ -> the coefficient of b, intercept_-> w
w_norm = sgdr.coef_
b_norm = sgdr.intercept_
print(f"model parameters: w: {w_norm}, b:{b_norm}")
print("normalised final w : [ 0.19005214  0.19903042 -0.07664191  0.18625677], b: 1.1481")
# making a prediction using sgdr.predict

res = sgdr.predict(x_test)
print(res)
print("vs Actual:")
print(y_train)

