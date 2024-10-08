# Machine Learning

>[!Note]
>This course starts from beginner and basics of ML and covers upto Intermediate level concepts.

#### So, **What is machine learning** ?
A field of study that gives the computers an ability to learn without being explicitly programmed.

#### **Types of Machine Learning:**
1. **Supervised Learning**
2. **Unsupervised Learning**
3. **Recommender Systems.**
4. **Reinforcement learning.**

### Terminology

- **Training Set -** Data set used to train the model $`_{targets}^{features}`$
- **Feature -** Standard notation to denote the input variable `x` in a data set.
- **target -** Standard notation to denote the output variable `y` in a data set.
- ***`m` -*** Number of training Examples (e.g. rows in a data set).
- ***$`(x^{(i)},y^{(j)})`$*** - Denotes the $i^{th}$ Feature and $j^{th}$ target of the training set as a training example (Basically used to denote the Feature and Target set being used as a training example).
    > Note: ***$`x^{2}`$*** does not mean ***square of `x`***, instead it means ***$`2^{nd}`$ Feature*** of the training set.
- ***$`f`$*** - Called the **Model** is the $Function$, that is produced after the training set data is fed into the Supervised training algorithm.
- ***$`\hat{y}`$*** - **Prediction** that was produced from the function $f$ for input of $x$.

### Notation
---
Here is a summary of some of the notations, updated for multiple features.  

| General Notation                  | Description                                                                                       | Python (if applicable) |
|:----------------------------------|:--------------------------------------------------------------------------------------------------|:-----------------------|
| $\alpha$                           | scalar, non bold                                                                                  |                        |
| $\overrightarrow{a}$                  | vector, bold                                                                                      |                        |
| ${A}$                  | matrix, bold capital                                                                              |                        |
| **Regression**                    |                                                                                                   |                        |
| $f{(X)}$                  | training example matrix                                                                           | `X_train`              |
| $y$                  | training example targets                                                                          | `y_train`              |
| $\overrightarrow{w}^{(i)}, y^{(i)}$   | \( i \)-th Training Example                                                                       | `X[i]`, `y[i]`         |
| $m$                           | number of training examples                                                                       | `m`                    |
| $n$                           | number of features in each example                                                                | `n`                    |
| $\overrightarrow{w}$                  | parameter: weight                                                                                 | `w`                    |
| $b$                           | parameter: bias                                                                                   | `b`                    |
| $\( f_{\overrightarrow{w},b}(\overrightarrow{w}^{(i)}) \)$ | The result of the model evaluation at $\( \overrightarrow{w}^{(i)} \)$ parameterized by $\( \overrightarrow{w},b \): \( f_{\overrightarrow{w},b}(\overrightarrow{w}^{(i)}) = \overrightarrow{w} \cdot \overrightarrow{w}^{(i)} + b \)$ | `f_wb`                 |



## Supervised Learning

Supervised learning is a type of machine learning where the model is trained on a labeled dataset. In this dataset, each input comes with a corresponding output or label. The model learns to map inputs to the correct outputs by finding patterns and relationships in the data. Once trained, the model can make predictions on new, unseen data.
In supervised learning the ML Algorithm is given an imput `x` and for every input of x a right answer or output of `y` is given. The ML Algorithm then tries to find out relationships and patterns so that for every new input `x` it can produce the correct output.

**Types:**
1. Regression
2. Classification



## 1. Regression

>[!Tip]
>In these types of models the predictions or outputs `y` can be infinite.

The model has to output a number after analyzing the input, it may produce infinitely possible outcomes.
Regression is a type of supervised learning used in machine learning and statistics to predict a continuous outcome variable based on one or more predictor variables. The goal of regression is to model the relationship between the dependent variable (the outcome we want to predict) and the independent variables (the predictors). Infinitely many outputs `y` that are possible are predicted.

### Linear Regression (Univariate Linear Regression)

Linear regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data. Using a single variable or input here in the regression model.
For the number of inputs `x` there should be the same number of outputs `y` in the sample data.

>[!Note]
> **Model $f$** ->  **$f_{w,b}$(x) = $wx+b$**, When $f$ is a straight line.

>[!Tip]
> $f$ is the function that makes prediction  **$`\hat{y}`$** based on **$_{w,b}$** which are numbers for the input feature $`x`$. <br>
>**Alternatively** or simply $f(x)$ = $wx+b$.

![Model](imgs/image.png)

The goal of the is to find a value of $w$ and $b$ such that $j_{(w,b)}$ - the cost function is at a minimum.

$`^{min}_{w,b}J_{(w,b)}`$

This way we will be able to arrive at the values of $w$ and $b$ that fits our training set really well.

Now in order to do this automatically we will use gradient Descent.
The goal of the gradient descent is to find a value of $w$ and $b$ such that $j_{(w,b)}$ is at a minimum.


**Model :** 

$$f_{w,b}(x^{(i)}) = wx^{(i)} + b$$


**Cost Function :** 

$$J_{(w,b)} = \frac{1}{2m}\sum_{i=0}^m (f_w,_b(x^{(i)})-y^{(i)})$$



**Gradient Descent:**

$$w = w - \alpha.\frac{\partial }{\partial w}J_{(w,b)}$$

$$b = b - \alpha.\frac{\partial }{\partial b}J_{(w,b)}$$



**Alternatively,**

$$w = w - \alpha . [\frac{1}{m}\sum_{i=0}^m ((f_w,_b(x^{(i)})-y^{(i)}).x^{(i)})]$$

$$b = b - \alpha . [\frac{1}{m}\sum_{i=0}^m (f_w,_b(x^{(i)})-y^{(i)})]$$



**Finally,**

> $$w = w - \alpha . [\frac{1}{m}\sum_{i=0}^m (((w.x^{(i)} +b )-y^{(i)}).x^{(i))}]$$
>
> $$b = b - \alpha . [\frac{1}{m}\sum_{i=0}^m ((w.x^{(i)} +b )-y^{(i)})]$$
>

---

**In code:**

Compute the derivative of $w$ and $b$:


$$\frac{\partial }{\partial w}J_{(w,b)}$$

$$\frac{\partial }{\partial b}J_{(w,b)}$$


**which is,**

$$\frac{1}{m}\sum_{i=0}^m[(f_w,_b(x^{(i)})-y^{(i)}).x^{(i)}]$$ 

$$\frac{1}{m}\sum_{i=0}^m [((w.x^{(i)} +b )-y^{(i)}).x^{(i)}]$$



The code should compute the derivates and update values of $w$ and $b$ at once consistently,
The updated values of $w$ and $b$ should not interfere with one another.

Here we are computing the derivative of $\frac{\partial }{\partial w}J_{(w,b)}$ and $\frac{\partial }{\partial b}J_{(w,b)}$

 ```python
sum_w,sum_b = 0.0,0.0
for i in range(m):    
    f_wb_x = (w*x_t[i] + b) - y_t[i]
    sum_w = sum_w + f_wb_x*x_t[i]
    sum_b = sum_b + f_wb_x
sum_w = sum_w/m
sum_b = sum_b/m
return sum_w,sum_b

 ```

After we need to update the values of $w$ and $b$ at the same time by reducing $\alpha$ of $\frac{\partial }{\partial w}J_{(w,b)}$ and $\frac{\partial }{\partial b}J_{(w,b)}$:

```python

dq,w,db,b = 0.,0.,0.,0.
for i in range(self.iter):
    dw,db = self.derivative("single",w,b)
    w = w - alpha*dw
    b = b - alpha*db
    if i%1000 == 0 :
        print(f"w : {w} , b : {b}")
return w,b

```

>[!Note]
> It is very important to note that the formula converted into code stays true to the its original meaning.
> This is applicable for all the **ML** Algorithms.

### Incorrect Approach (Univariate Linear Regression):

The following approach may look like the updates of w and b are happening simultaneously but its is not the case.

>[!Tip]
> The derivative is a function that calculates the $\frac{\partial}{\partial w}J_{(w,b)}$ for w and b

```python
tempw, w, tempb, b = 0., 0., 0., 0.
for i in range(self.iter):
    tempw, tempb = self.derivative("single", w, b)
    tempw = tempw - alpha * tempw
    tempb = tempb - alpha * tempb
    w = tempw
    b = tempb
```

The above approach roughly translates to `tempw = ∂L/∂w - alpha * ∂L/∂w = ∂L/∂w * (1 - alpha)` or,

$$\frac{\partial}{\partial w}J_{(w,b)} - \alpha . \frac{\partial}{\partial w}J_{(w,b)}$$

**But**,

the actual formula is: 

$$w -\alpha . \frac{\partial}{\partial w}J_{(w,b)}$$


### Multivalue Linear Regression

>[!Note]
> When we say the cost function is being calculated we are not referring to the ***Squared value cost function*** here.
> Instead we are talking about the section of the formula which was derived from the fromula of gradient descent after solving it.
> Gradient descent formula for w:
> $$\frac{\partial J(\overrightarrow{w},b)}{\partial w_j}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} f_{\overrightarrow{w},b}(\overrightarrow{w}^{(i)}) - y^{(i)})x_{j}^{(i)}$$
> **vs**
> Squared value cost function as we initially saw:
> $$J_{(\overrightarrow{w},b)} = \frac{1}{2m}\sum_{i=0}^m (f_{\overrightarrow{w},b}(x^{(i)})-y^{(i)})^2$$

Using multiple values or features in Linear Regression model.

---
| Size (sqft) | Number of Bedrooms  | Number of floors | Age of  Home | Price (1000s dollars)  |   
| ----------------| ------------------- |----------------- |--------------|-------------- |  
| 2104            | 5                   | 1                | 45           | 460           |  
| 1416            | 3                   | 2                | 40           | 232           |  
| 852             | 2                   | 1                | 35           | 178           |  

---
 You will build a linear regression model using these values so you can then predict the price for other houses. For example, a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old.


X is no longer a single feature in this types of regression models. Instead x is a set of features.
Y is the target.

```python

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

```
### Parameter vector w, b

* $\overrightarrow{w}$ is a vector with $n$ elements.
  - Each element contains the parameter associated with one feature.
  - in our dataset, n is 4.
  - notionally, we draw this as a column vector

$$\overrightarrow{w} = \begin{pmatrix}
w_0 \\ 
w_1 \\
\cdots\\
w_{n-1}
\end{pmatrix}
$$
* $b$ is a scalar parameter.  


### Matrix of x Features



$`
\overrightarrow{x} = \begin{pmatrix}
x^{(0)}_0 & x^{(0)}_1 & \cdots & x^{(0)}_{n-1} \\
x^{(1)}_0 & x^{(1)}_1 & \cdots & x^{(1)}_{n-1} \\
\vdots & \vdots & \ddots & \vdots \\
x^{(m-1)}_0 & x^{(m-1)}_1 & \cdots & x^{(m-1)}_{n-1}
\end{pmatrix}
`$




notation:

- $\overrightarrow{w}^{(i)}$ is vector containing example i. $`\overrightarrow{w}^{(i)} = (x^{(i)}_0, x^{(i)}_1, \cdots , x^{(i)}_{n-1})`$
- $x^{(i)}_j$ is element j in example i. The superscript in parenthesis indicates the example number while the subscript represents an element. 

---

### Model Prediction With Multiple Variables

The model's prediction with multiple variables is given by the linear model:

$$f_{(\overrightarrow{w},b)}(\overrightarrow{x}) =  w_0x_0 + w_1x_1 + \cdots + w_{n-1}.x_{n-1} + b$$
or in vector notation:
$$f_{\overrightarrow{w},b}(\overrightarrow{x}) = \overrightarrow{w} \cdot \overrightarrow{x} + b$$ 
where $\cdot$ is a vector `dot product`

### Gradient Descent With Multiple Variables

Gradient descent for multiple variables:

$$\begin{align*} \text{repeat}&\text{ until convergence:} \ \lbrace \newline\
& w_j = w_j -  \alpha \frac{\partial J(\overrightarrow{w},b)}{\partial w_j} \tag{5}  \; & \text{for j = 0..n-1}\newline
&b\ \ = b -  \alpha \frac{\partial J(\overrightarrow{w},b)}{\partial b}  \newline \rbrace
\end{align*}$$

where, n is the number of features, parameters $w_j$,  $b$, are updated simultaneously and where  

$$\begin{align}\frac{\partial J(\overrightarrow{w},b)}{\partial w_j} = \frac{1}{m} \sum\limits_{i = 0}^{m-1} f_{\overrightarrow{w},b}(\overrightarrow{w}^{(i)}) - y^{(i)})x_{j}^{(i)} \tag{6}  \\
\frac{\partial J(\overrightarrow{w},b)}{\partial b}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\overrightarrow{w},b}(\overrightarrow{w}^{(i)}) - y^{(i)})\tag{7}\end{align}$$
* m is the number of training examples in the data set

    
*  $f_{\overrightarrow{w},b}(\overrightarrow{w}^{(i)})$ is the model's prediction, while $y^{(i)}$ is the target value


### The Code

>[!Note]
> $w$ is a vector of size $n$ and $n$ is the number of features or columns in the ***training set X_train***.
> $b$ is a scalar.
> $J_{(w,b)}$ is a scalar as well.

The gradient of the cost function needs to be calculated by  
$$\frac{1}\{m}\sum_{i=0}^m [((\overrightarrow{w}.\overrightarrow{x}^{(i)} +b )-y^{(i)}).x^{(i)}_j]$$

So the values of $w_1$ all the way up till $w_n$ for $x^{(i)}_1$ to $x^{(i)}_n$ and summed.

In simpler terms all the features of the training set ***X_train*** is being summed from 1 ... n for w[1]...w[n]

### **Implementing**


The algorithm has three parts here as well roughly, 

- **The cost function or error calculation** -  which in terms of the gradient descent algorithm translates to the derivative section of the algorithm, except that for multivalue linear regression models its a dot product of the vectors $(\overrightarrow{w}.\overrightarrow{x}^{(i)} +b )$.

- **The learning algorithm of Gradient descent** - The value of $w$ and $b$ is deducted from itelf times the learning rate $\alpha$.

- **The value prediction** Using the multivalue linear regression model to calculate the predicted value.

#### **Calculating the Derivative section of Gradient Descent Algorithm**
>[!Tip]
>I have found it to be often easier to split the Algorithm equation into segments and process those segments as different functions before tying the results together and completing the equation.
> For example the operation of dot product of the vectors of $w$ and $x^{(i)}$ and the $\sum$ operation of the derivative function (${\frac{\partial J}{\partial w}}$) of the gradient descent algorithm.

```python

for i in range(m):
    j_wb = np.dot(w,x_t[i]) + b - y_t[i]
    db = db + j_wb
    for j in range(n):
        dw[j] = dw[j] + j_wb*x_t[i,j]
return dw/m,db/m

```

**Explaination**

- This will give us the derivative of the gradient descent algorithm.
- Here b is being calculated as usual since b is a scalar.
- **j_wb is the error margin that is being calculated by a dot product of the vectors of $w$ and $x$.**
- `dw[j] = dw[j] + j_wb*x_t[i,j]` **is the equivalent of $[((\overrightarrow{w}.\overrightarrow{x}^{(i)} +b )-y^{(i)}).x^{(i)}_j]$**
- `return dw/m,db/m` returns the derivative of the gradient descent after calculating the mean.

>[!Tip]
> x_t[i] is a vector and a subset of X_train, the reason its a vector is because its a row of the 2D array X_train which is an array of size $n$.


#### **The learning algorithm of Gradient descent**

```python

for i in range(self.iterations):
    dw,db = self.compute_gradient(w,b)
    w = w - alpha*dw
    b = b - alpha*db
print("learn_gradient: ",w,b)
return w,b

```

**Explaination**


>[!Note]
>The equation is:
>
> $\frac{\partial J_{(\overrightarrow{w},b)}}{\partial w_j} = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\overrightarrow{w},b}(\overrightarrow{x}^{(i)}) - y^{(i)}) x_{j}^{(i)} \tag{6}$


- `dw` and `db` are the values returned after calculating the derivatives.
- Then the values of `w` and `b` are updated by the learning rate $\alpha$.
- Since `w` and `b` are vectors, the vector multiplication and vector substraction is happening here`w = w - alpha*dw`.
- Finally the values of w and b are returned.
- This stays true to the original equation.


#### **The value prediction**

>[!Note]
>The equation
> $\overrightarrow{w}.\overrightarrow{x}^{(i)} +b$

```python

p = np.dot(w,x_in) + b
return p

```


>[!Tip]
> When we say the cost function is being calculated we are not referring to the initial ***Squared value cost function*** here.
> Instead we are talking about the section of the formula which finds the cost or **error margin** within the gradient descent algorithm formula after solving it.
> Gradient descent formula for w:
> $\frac{\partial J_{(\overrightarrow{w},b)}}{\partial w_j}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\overrightarrow{w},b}(\overrightarrow{w}^{(i)}) - y^{(i)})x_{j}^{(i)}$
> **vs**
> Squared value cost function as we initially saw:
> $$J_{(\overrightarrow{w},b)} = \frac{1}{2m}\sum_{i=0}^m (f_{\overrightarrow{w},b}(x^{(i)})-y^{(i)})^2$$
> **The cost function we are referring to here:**
>$$(f{\overrightarrow{w},b}(\overrightarrow{w}^{(i)}) - y^{(i)})$$


## Feature Scaling

Re-scaling features helps optimize the gradient algorithm, if the minimum and maximum values of the feature sets are too far apart, it will make the learning algorithm in our case the **gradient descent** algorithm to take longer to find the minimum of $j_{(w,b)}$


Consider the training set below:

--
| Size (sqft) x1| Number of Bedrooms x2 | Price (1000s dollars) y  |   
| ----------------| ------------------- |-------------- |  
| 2104            | 5                   | 460           |  
| 1416            | 3                   | 232           |  
| 852             | 2                   | 178           |  

---

**Here,**

If we were to plot the points in a scatter graph for $x axis at x_1$ and $y axis at x_2$ for $w_1$ and $w_2$ the cost function of the Gradient Descent : $J_{(\overrightarrow{w},b)}$ which is a calculated by:
Lets assume $w_1 = 50 , w_2 = 10$   
**Then,** the scatter plot would be heavily focused on the $x_1$ axis and the **contour plot** would be a narrow elipse for $x_1$ and $x_2$ since $w_1$ would have to be a smaller value to impact the overall $x_1.w_1$ whereas $w_2$ can to be for $x_2.w_2$.

**Why use Feature scaling?**

- It helps to optimize Gradient Descent making it faster to reach the $`^{min}_{w,b}J_{(w,b)}`$
- If left unscaled it will produce a graph where the values of a feature is heavily focused as compared to the others.
- Gradient Descent runs faster.

Which brings to our next agenda - **How do we scale features?**

### Max Normaliaztion

- Dividing each max value of features $max(x_1)$ to the original $x_1$.

$$\frac{x_1}{max(x_1)}$$

**Example,**

For the above example - $x_1$ = `[2104,1416,852]`, $x_2$ = `[5,3,2]`

applying the formula both features $\frac{x_1}{max(x_1)} = \frac{2104}{2104} , \frac{1416}{2104} ,\frac{852}{2104}$
scaled $x_1$ = `[1,0.66,0.4]`
scaled $x_2$ = `[1,0.6,0.4]`

If we were to plot the rescaled values of $x_1$ and $x_2$ we would get a more uniform scatter plot and a more circular plot for $w_1$ and $w_2.

### Mean Normalization

- Find the average of each feature set $x_1$ to $x_n$ called $\mu$.
- Substract $\mu$ from all values of feature set of $x_1$ and divide it by $max(x_1)-min(x_1)$
- This will give a scalar in range of positive to negative.

$$\frac{x_1 - \mu _1}{max(x_1)-min(x_1)}$$

>**The advantage this scaling of features brings is a noticable reduction in the time taken by the algorithm of Gradient descent to Converge.**

### Z-Score Normalization

- Find mean of each **column or feature** of the training set.
- Find the standard deviation of each **column/feature** of the training set.
- element-wise, subtract mu for that column from each example, divide by standard deviation for that column

```python
# find the mean of each column/feature
mu     = np.mean(self.x_t, axis=0)                 # mu will have shape (n,)
# find the standard deviation of each column/feature
sigma  = np.std(self.x_t, axis=0)                  # sigma will have shape (n,)
# element-wise, subtract mu for that column from each example, divide by std for that column
X_norm = (self.x_t - mu) / sigma

```

>[!Note]
>**Each new set of inputs or features for which the value needs to be predicted `x_in` of shape n for `x_train` of size $mxn$ needs to be rescaled to fit the *normalized* values of `x_train`**
> Lets say if ***Z-score-Normalization*** has been used to re-scale the values of `x_train` then the same values of `sigma` and `mu` that was generated while performing the normalization needs to be used to re-scale the values of `x_in`.


## Scikit-Learn

- **`scikit-learn`** is a free and open-source machine learning library for the Python programming language.
- Through scikit-learn, we can implement various machine learning models for regression, classification, clustering, and statistical tools for analyzing these models.,

So all the models,cost-function and learning algorithms that we have implemented from scratch for all the different types of *Regression* models as well as, all the other models that we are going to see moving forward can be implemented using **`scikit-learn`**.

### Installing `sklearn`

```python
py -m pip install scikit-learn

```
or

for python version < `3.8`

```python
pip install scikit-learn

```

### Using `sklearn`

```python
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

```

`SGDRegressor` is a Linear model fitted by minimizing a regularized empirical loss with SGD.
SGD stands for Stochastic Gradient Descent: the gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule (aka learning rate).
In simple terms it enables the dev to implement the **Gradient Descent** algorithm and **squared value cost function** for **Linear Regression** models that we are familiar with.

`StandardScaler` lets one implement ***z-score normalization*** for scaling the features if the value disparity is too large.

>[!Note]
>The Multivalue Linear Regression is being implemented by `sklearn` here. The model and learning algorithm are the same as what we have used so far. It just provides a better **framework** and less code to implement the same.

**Scaling using `sklearn`**
- This is how the features of training set `X_train` are normalized using `sklearn` implementing *z-score normalization**

```python
scaler = StandardScaler()
x_t_scaled = scaler.fit_transform(X_train)

```

**Gradient Descent using `sklearn`** 
- This following code implements the Gradient Descent algoritm and generates the values of `w` and `b`.
- Iterations will continue untill the $`^{min}_{\overrightarrow{w},b}J_{(\overrightarrow{w},b)}`$ is reached and the Gradient Descent algorithm converges.

```python
sgdr = SGDRegressor(max_iter=10000)
sgdr.fit(x_t_scaled,y_train)
print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")

```

**Prediction using `sklearn`** 

- $w$ and $b$ can be fetched after the `sklearn` model converges and then can be used to be applied to the lineear regression model $f_{w,b}$(x) = $wx+b$.

```python
w_norm = sgdr.intercept_
b_norm = sgdr.coef_
print(f"model parameters: w: {w_norm}, b:{b_norm}")

```

- Or one can simply use the `predict()` function to make the prediction

```python
res = sgdr.predict(x_t_scaled)
print(res)

```

## Feature Engineering

Using intuition and experience to design new features from existing features by transforming or combining them.

![Model](imgs/feature-eng.png)



## 2. Classification

>[!Tip]
>This model will only have specific outputs. These outputs will be labelled/Classified 
Model will only have Specific outputs. These outputs will be labelled/Classified into classification.

Note:

```
Table
Size     : CM
Diagnosis: Malignant (1) or Benign (0)
+---------------+-----------+
| Size (x)      | Diagnosis |
|               |   (y)     |
|---------------|-----------|
|   2           |   0       |
|   5           |   1       |
|   1           |   0       |
|   7           |   1       |
+---------------+-----------+
```

>The table above shows the data for patients being diagnosed for breast cancer with lumps in cms,the resulting diagnosis that was done for each of these sizes. 
>The Diagnosis is either Benign or Malignant - These will be our classification, now according to the inputs in size our **ML model will be able to predict whether the diagnosis will be either of these two classifications**



- In classification the terms ***class*** and ***category*** are used interchangeably.
- More than one inputs can be used. 
- Prediction is either Positive `1` or Negative `0`.
- When the problem does not aim to produce a numerical prediction but in terms of **yes** or **no**.

### Why not use Linear Regression for Classification Problems?

Although it is definitely possible to use Linear Regression to make a prediction but Linear Regression produces a scalar prediction, in terms of number.


![Model](imgs/classification-lr.png)



- Lets plot the `x_train` and `y_train` for the example above. The target is `y_train` which is 0 or 1, and the feature(s) are the sizes.
- When plotted, the graph looks like as depicted above.
- Now If we try to fit a straight line through the data trying to predict using Gradient Descent and linear Regresession Model we get a graph like ablove. *Blue line depicts the prediction graph*.
- As shown in the figure above we can draw a decision making threshold from y-axis at lets say `0.5` now we can decalare that any values above this threshold is a **yes** and below it is **no**.
- This creates a *Decision Boundary* like the blue intersecting line trhough the curve. Anything to the left of this line is a **no** or `0` and everything to the right is **yes** or `1`.
- Gradually as we keep increasing the data we find the graph shifts towards the right developing a less steep slope, which ultimately leads to the *decision boundary* towards to right as well.
- This is what causes the false predictions to occur.


## Logistics Regression

Logistic regression is a supervised machine learning algorithm that accomplishes binary classification tasks by predicting the probability of an outcome, event, or observation. The model delivers a binary or dichotomous outcome limited to two possible outcomes: yes/no, 0/1, or true/false.

- The model uses sigmoid function to draw a `s-shaped` curve fitting all the training data and helps in decision making of whether the prediction should be `0 false` or `1 true`.


---

## *Un*supervised Learning
>[!Tip]
> The Algoritm is allowed to figure out useful or interesting data on its own from a given data set,
> we are not expecting it to have correct/specific answers by labelling the data and supervising it.

Also caused clustering, is a type of lerning algorithms which is not provided with **labelling** of the outputs `y` for every inputs `x` like it was the case for supervised learning algorithms.
In this case the algorithm is allowed to learn or figure out on its own ***something interesting*** from a given data set. Since the data set isn't properly labelled the algoritm may try to come up with any relations that it can find in order to group the datas into **clusters**.

**Types**
1. Clustering - Group similar data together.
2. Anomaly Detection - Find unusual data points.
3. Dimensionality Reduction - Compress data using fewer numbers.

