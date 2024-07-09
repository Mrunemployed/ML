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

## 1.3 Notation
Here is a summary of some of the notation you will encounter, updated for multiple features.  

|General <img width=70/> <br />  Notation  <img width=70/> | Description<img width=350/>| Python (if applicable) |
|: ------------|: ------------------------------------------------------------||
| $a$ | scalar, non bold                                                      ||
| $\mathbf{a}$ | vector, bold                                                 ||
| $\mathbf{A}$ | matrix, bold capital                                         ||
| **Regression** |         |    |     |
|  $\mathbf{X}$ | training example matrix                  | `X_train` |   
|  $\mathbf{y}$  | training example  targets                | `y_train` 
|  $\mathbf{x}^{(i)}$, $y^{(i)}$ | $i_{th}$Training Example | `X[i]`, `y[i]`|
| m | number of training examples | `m`|
| n | number of features in each example | `n`|
|  $\mathbf{w}$  |  parameter: weight,                       | `w`    |
|  $b$           |  parameter: bias                                           | `b`    |     
| $f_{\mathbf{w},b}(\mathbf{x}^{(i)})$ | The result of the model evaluation at $\mathbf{x^{(i)}}$ parameterized by $\mathbf{w},b$: $f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = \mathbf{w} \cdot \mathbf{x}^{(i)}+b$  | `f_wb` | 


## Supervised Learning

Supervised learning is a type of machine learning where the model is trained on a labeled dataset. In this dataset, each input comes with a corresponding output or label. The model learns to map inputs to the correct outputs by finding patterns and relationships in the data. Once trained, the model can make predictions on new, unseen data.
In supervised learning the ML Algorithm is given an imput `x` and for every input of x a right answer or output of `y` is given. The ML Algorithm then tries to find out relationships and patterns so that for every new input `x` it can produce the correct output.

**Types:**
1. Regression
2. Classification

---

### 1. Regression

>[!Tip]
>In these types of models the predictions or outputs `y` can be infinite.

The model has to output a number after analyzing the input, it may produce infinitely possible outcomes.
Regression is a type of supervised learning used in machine learning and statistics to predict a continuous outcome variable based on one or more predictor variables. The goal of regression is to model the relationship between the dependent variable (the outcome we want to predict) and the independent variables (the predictors). Infinitely many outputs `y` that are possible are predicted.

## Linear Regression (Univariate Linear Regression)

Linear regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data. Using a single variable or input here in the regression model.
For the number of inputs `x` there should be the same number of outputs `y` in the sample data.

>[!Note]
> **Model $f$** ->  **$f_{w,b}$(x) = $wx+b$**, When $f$ is a straight line.

>[!Tip]
> $f$ is the function that makes prediction  **$`\hat{y}`$** based on **$_{w,b}$** which are numbers for the input feature $`x`$. <br>
>**Alternatively** or simply $f(x)$ = $wx+b$.

![Model](image.png)

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

In code:

 Compute the derivative of:
 for w $$\frac{\partial }{\partial w}J_{(w,b)}$$ and,
 for b $$\frac{\partial }{\partial b}J_{(w,b)}$$
 which is 
 $$((f_w,_b(x^{(i)})-y^{(i)}).x^{(i)})$$ for w,
 $$[\frac{1}{m}\sum_{i=0}^m ((w.x^{(i)} +b )-y^{(i)})]$$ for b.

 the code should compute the derivates and update values of $w$ and $b$ at once consistently,
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
After we need to update the values of $w$ and $b$ at the same time by reducing $\alpha$ * to $\frac{\partial }{\partial w}J_{(w,b)}$ and $\frac{\partial }{\partial b}J_{(w,b)}$:

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

### Incorrect Approach:

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

$$ w -\alpha . \frac{\partial}{\partial w}J_{(w,b)}$$


#### Multivalue Linear Regression

Using multiple values or features in Linear Regression model.


---


### 2. Classification

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

