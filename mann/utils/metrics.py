import theano
import theano.tensor as T
import numpy as np


# predictions is the argmax of the posterior
def accuracy_instance(predictions, targets, n=[1, 2, 3, 4, 5, 10], \
        nb_classes=5, nb_samples_per_class=10, batch_size=1):
    accuracy_0 = theano.shared(np.zeros((batch_size, nb_samples_per_class), \
        dtype=theano.config.floatX))
    indices_0 = theano.shared(np.zeros((batch_size, nb_classes), \
        dtype=np.int32))
    batch_range = T.arange(batch_size)
    def step_(p, t, acc, idx):
        acc = T.inc_subtensor(acc[batch_range, idx[batch_range, t]], T.eq(p, t))
        idx = T.inc_subtensor(idx[batch_range, t], 1)
        return (acc, idx)
    (raw_accuracy, _), _ = theano.foldl(step_, sequences=[predictions.dimshuffle(1, 0), \
        targets.dimshuffle(1, 0)], outputs_info=[accuracy_0, indices_0])
    accuracy = T.mean(raw_accuracy / nb_classes, axis=0)

    return accuracy