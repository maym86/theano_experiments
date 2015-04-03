__author__ = 'Michael May'

#http://deeplearning.net/software/theano/tutorial/examples.html

from theano.tensor.shared_randomstreams import RandomStreams
from theano import function

srng = RandomStreams(seed=234)
rv_u = srng.uniform((2,2))
rv_n = srng.normal((2,2))
f = function([], rv_u)
g = function([], rv_n, no_default_updates=True)    #Not updating rv_n.rng
nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)

#Call the random number function - normally distributed
print f()
print f()

#same value every time - no_default_updates = True
print g()  # different numbers from f_val0 and f_val1
print g()

#Seeding streams

rng_val = rv_u.rng.get_value(borrow=True)   # Get the rng for rv_u
rng_val.seed(89234)                         # seeds the generator
rv_u.rng.set_value(rng_val, borrow=True)    # Assign back seeded rng

srng.seed(902340)  # seeds rv_u and rv_n with different seeds each

state_after_v0 = rv_u.rng.get_value().get_state()
nearly_zeros()       # this affects rv_u's generator
v1 = f()
rng = rv_u.rng.get_value(borrow=True)
rng.set_state(state_after_v0)
rv_u.rng.set_value(rng, borrow=True)
v2 = f()             # v2 != v1
v3 = f()             # v3 == v1

print v1, v2, v3