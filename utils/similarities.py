import theano.tensor as T

def cosine_similarity(x, y, eps=1e-6):
    z = T.dot(x, y.T)
    z /= T.sqrt(T.outer(T.sum(x * x, axis=1), T.sum(y * y, axis=1)) + eps)

    return z
