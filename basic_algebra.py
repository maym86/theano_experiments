__author__ = 'Michael May'

#http://deeplearning.net/software/theano/tutorial/adding.html
from theano import *
import theano.tensor as T
import numpy as np

#pretty print for theano
from theano import pp

#simple algebra
x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x, y], z)

#show the function z
print pp(z)
print f(2, 3)
print f(16.3, 12.1)

#matrix multiplication
x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
f = function([x, y], z)

#apply to list
print f([[1, 2], [3, 4]], [[10, 20], [30, 40]])

#apply to numpy array
print f(np.array([[1, 2], [3, 4]]), np.array([[10, 20], [30, 40]]))

#exercise 1
a = theano.tensor.vector() # declare variable
b = theano.tensor.vector() # declare variable
out = a ** 2 + b ** 2 + 2 * a * b # build symbolic expression
f = theano.function([a, b], out)   # compile function
print f([0, 1, 2],[3, 4, 5])  # prints result



