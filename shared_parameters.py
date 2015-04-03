__author__ = 'Michael May'

#http://deeplearning.net/software/theano/tutorial/examples.html
from theano import *
import theano.tensor as T
from theano import shared

#accumulator function with a shared variable
state = shared(0)
inc = T.iscalar('inc')
accumulator = function([inc], state, updates=[(state, state+inc)])
#get the state of the global variable
print state.get_value()
accumulator(1)
print state.get_value()
accumulator(300)
print state.get_value()

#set the value
state.set_value(-1)
accumulator(3)
print state.get_value()

#decrementor
decrementor = function([inc], state, updates=[(state, state-inc)])
decrementor(2)
print state.get_value()

#Using the function an dskipping the shared variable for a particular usage

fn_of_state = state * 2 + inc
# The type of foo must match the shared variable we are replacing
# with the ``givens``
foo = T.scalar(dtype=state.dtype)
skip_shared = function([inc, foo], fn_of_state,givens=[(state, foo)])
print skip_shared(1, 3)  # we're using 3 for the state, not state.value
print state.get_value()  # old state still there, but we didn't use it
