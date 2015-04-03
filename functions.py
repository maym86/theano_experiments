__author__ = 'Michael May'

#http://deeplearning.net/software/theano/tutorial/examples.html


from theano import *
import theano.tensor as T

#Logistic Function
x = T.dmatrix('x')
s = 1 / (1 + T.exp(-x))
logistic = function([x], s)
print logistic([[0, 1], [-1, -2]])

#More than one thing at the same time
a, b = T.dmatrices('a', 'b')
diff = a - b
abs_diff = abs(diff)
diff_squared = diff**2
f = function([a, b], [diff, abs_diff, diff_squared])

#three outputs from one input
print f([[1, 1], [1, 1]], [[0, 1], [2, 3]])

#Setting a Default Value for an Argument
from theano import Param
x, y = T.dscalars('x', 'y')
z = x + y
f = function([x, Param(y, default=1)], z)
print f(33) #default is 1
print f(33, 2) #try new value

#named parameters
x, y, w = T.dscalars('x', 'y', 'w')
z = (x + y) * w
f = function([x, Param(y, default=1), Param(w, default=2, name='w_by_name')], z)
print f(33)
print f(33, 2)
print f(33, 0, 1)
print f(33, w_by_name=1)
print f(33, w_by_name=1, y=0)
