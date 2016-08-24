import theano.tensor as T

def cosine_similarity(x, y, eps=1e-6):
    z = T.batched_dot(x, y.dimshuffle(0, 2, 1))
    z /= T.sqrt(T.sum(x * x, axis=2).dimshuffle(0, 1, 'x') * T.sum(y * y, axis=2).dimshuffle(0, 'x', 1) + eps)

    return z
