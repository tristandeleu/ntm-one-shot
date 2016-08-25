import theano
import numpy as np

def shared_glorot_uniform(shape, dtype=theano.config.floatX, name='', n=None):
    if isinstance(shape, int):
        high = np.sqrt(6. / shape)
    else:
        high = np.sqrt(6. / (np.sum(shape[:2]) * np.prod(shape[2:])))
    shape = shape if n is None else (n,) + shape
    return theano.shared(np.asarray(
        np.random.uniform(
            low=-high,
            high=high,
            size=shape),
        dtype=dtype), name=name)

def shared_zeros(shape, dtype=theano.config.floatX, name='', n=None):
    shape = shape if n is None else (n,) + shape
    return theano.shared(np.zeros(shape, dtype=dtype), name=name)

def shared_one_hot(shape, dtype=theano.config.floatX, name='', n=None):
    shape = (shape,) if isinstance(shape, int) else shape
    shape = shape if n is None else (n,) + shape
    initial_vector = np.zeros(shape, dtype=dtype)
    initial_vector[..., 0] = 1.
    return theano.shared(initial_vector, name=name)

def weight_and_bias_init(shape, dtype=theano.config.floatX, name='', n=None):
    return (shared_glorot_uniform(shape, dtype=dtype, name='W_' + name, n=n), \
            shared_zeros((shape[1],), dtype=dtype, name='b_' + name, n=n))