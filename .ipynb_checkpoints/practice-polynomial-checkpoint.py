import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import re

def get_this(name):
    try:    
        exp = re.compile(f"[{name}]*")
        dirc = os.path.abspath(os.path.pardir)
        path = os.path.join(dirc,"ML","datasets")
        lsdir = os.listdir(path)
        file = [x for x in lsdir if re.search(exp,x)]
        return os.path.join(path,file[0])
    except Exception as err:
        return err
    
file = get_this('student')
df = pd.read_csv(file)
print(df.describe())
df.drop_duplicates(inplace=True)
# df['Extracurricular Activities'] = np.where(df['Extracurricular Activities'] == "Yes", 1,0)
df.drop(columns='Extracurricular Activities')
print(df.columns)
_train = df.sample(frac=0.8,random_state=200)
_test = df.drop(_train.index)
print(df.shape)
sns.pairplot(df)

X_train = _train.iloc[:,:-1].values
y_train = _train['Performance Index'].values
x_test = _test.iloc[:,:-1].values

# X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
# y_train = np.array([460, 232, 178])
# for i in range(X_train.shape[1]):
#     plt.scatter(X_train[:,i],y_train, marker='o')
# plt.plot()
# plt.xlabel('x_train')
# plt.ylabel('Performance')
plt.show()