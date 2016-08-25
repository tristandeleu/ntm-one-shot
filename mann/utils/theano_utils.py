import theano
import numpy as np

def shared_floatX(x, name=''):
    return theano.shared(np.asarray(x, dtype=theano.config.floatX), name=name)