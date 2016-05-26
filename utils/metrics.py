import theano
import theano.tensor as T
import numpy as np


# predictions is the argmax of the posterior
def accuracy_instance(predictions, targets, n=[1, 2, 3, 4, 5, 10], \
        nb_classes=5, nb_samples_per_class=10):
    accuracy_0 = theano.shared(np.zeros(nb_samples_per_class, \
        dtype=theano.config.floatX))
    indices_0 = theano.shared(np.zeros(nb_samples_per_class, dtype=np.int32))
    def step_(p, t, acc, idx):
        acc = T.inc_subtensor(acc[idx[t]], T.eq(p, t))
        idx = T.inc_subtensor(idx[t], 1)
        return (acc, idx)
    (raw_accuracy, _), _ = theano.foldr(step_, sequences=[predictions, targets], \
        outputs_info=[accuracy_0, indices_0])
    accuracy = raw_accuracy / nb_classes

    return accuracy