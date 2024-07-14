import numpy as np
import random

# Declaring a simple numpy array
a = np.array([1,4,5,1,2,8])
#print(f"a = {a}")

# Creating a numpy array with arange(start,end)
b = np.arange(1.,10.)
#print(f"a = {b}")

# Creating a numpy 2d array filled with zeros of size (m*n) m-> rows, n-> columns.
# Each row of the said array is a vector or a list of numbers (scalars)
# to do this we pass in a tuple ((m*n))
c = np.zeros((2,3))
# or

# creating a 1d array
d = np.zeros(5)

#print(f"c = \n {c}")
#print(f"d = {d}")

# random.random_sample will create a 1D or 2D array and fill it with random float values depending on what shape is being passed as an argument
e = np.random.random_sample(10)
#print(f"e = {e}")

# depending on what shape is being passed as an argument, if a tuple of (m,n) is passed it will create a 2D Array
f = np.random.random_sample((3,3))
#print(f"f = \n {f}")

# Vectors (list of numbers) in Numpy -  addition, dot function, multiplication etc..
import time

def vector_dot(a:np.ndarray[float],b:np.ndarray[float]):
    if a.shape == b.shape:
        s = 0
        for i in range(a.shape[0]):
            s+= a[i]*b[i]
        return s
    else:
        return False

vec1 = np.random.rand(100000)
vec2 = np.random.rand(100000)

# First we will be using the vector dot method present in numpy out of the box
# Dot = sum of product between the pairs of numbers in each vector.
# => vec1[1]*vec2[1] + vec1[2]*vec2[2] + .... vec1[n]*vec2[n]
start = time.time()
vec3 = np.dot(vec1,vec2)
stop = time.time()
#print(f"time taken for np.dot : {(stop-start):.4f} ms")
#print(f"Vectorized dot: {vec3}")

# Now using our dot method:
start = time.time()
vec3 = vector_dot(vec1,vec2)
stop = time.time()
#print(f"time taken for our dot : {(stop-start):.4f} ms")
#print(f"Vectorized dot: {vec3}")

del(vec1)
del(vec2)

vec1 = np.arange(0,10)
vec2 = np.arange(10,20)

#print(f"vec1: {vec1} \nvec2: {vec2}")

# Multiplication

vec5 = vec1*2
#print(f"Multiplied the entire array with a scalar: {vec5}")

# Adding two vectors

vec3 = vec1+ vec2
#print(f"Adding the paired elements of a vector: {vec3}")

vec4 = vec1*vec2
#print("Multiplication of two vectors: ",vec4)

del(vec1)
del(vec2)
# Working with 2d arrays we can use the same approach - each row of a 2d array is a vector.

vec1 = np.random.random_sample((5,5))
vec2 = np.random.random_sample((5,5))
#print(vec1,"\n---------\n",vec2)
vec3 = np.dot(vec1,vec2)
#print("2D Array DOt vector:\n",vec3,"\n ---------\n")
# this is not exaclty dot of vectors anymore, instead numpy does a matrix multiplication, to dot properly - 
# #print(vec1.shape[0])
ls = np.zeros(vec1.shape[0])
#print(ls.shape)
for i in range(vec1.shape[0]):
    ls[i] = np.dot(vec1[i],vec2[i])
#print("np.dot: ",ls)


del(vec1,vec2,vec3,vec5,vec4)

# Reshape is a convinient way to create matrices

a = np.arange(12).reshape(4,3)
#print(a)

# Slicing Numpy arrays -> Same as slicing of lists or strings,
# Except the slicing here will not make a copy of the sliced values, insted it creates as view that references back to the original array.
M = np.random.random_sample((3,4))
#print("M: \n",M)
#print("M slicing: \n",M[0, ::3])
#print(f"M slicing - M[:, 1:3:1] {M.shape}: \n",M[:, 1:3:1])

m = np.arange(0,10)
c = m[::2]
#print("Before chaning m, c:",c)
m[4] = 21 
#print("After chaning m, c:",c)

# vs
m = list([1,2,3,4,5,6,7,8,9])
c = m[::2]
#print("Before chaning m, c:",c)
m[0] = 43
#print("After chaning m, c:",c)

m = np.arange(1,10)
p = np.arange(11,20)
m = m - 2*p
print(m)