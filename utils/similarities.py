import theano.tensor as T

def cosine_similarity(x, y, eps=1e-6):
    z = T.dot(y, x)
    z /= T.sqrt(T.sum(x * x) * T.sum(y * y, axis=1) + eps)

    return z