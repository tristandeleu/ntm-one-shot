import pytest

import theano
import theano.tensor as T
import numpy as np


def test_accuracy_instance():
    from metrics import accuracy_instance

    predictions_var, targets_var = T.imatrices('predictions', 'targets')
    accuracy_var = accuracy_instance(predictions_var, targets_var, \
        nb_classes=5, nb_samples_per_class=10, batch_size=16)
    accuracy_fn = theano.function([predictions_var, targets_var], accuracy_var)

    # Generate sample data
    targets = np.kron(np.arange(5), np.ones((16, 10))).astype('int32')
    predictions = np.zeros((16, 50)).astype('int32')

    indices = np.zeros((16, 5)).astype('int32')
    accuracy = np.zeros((16, 10))

    for i in range(16):
        for j in range(50):
            correct = np.random.binomial(1, 0.5)
            predictions[i, j] = correct * targets[i, j] + \
                (1 - correct) * ((targets[i, j] + 1) % 5)
            accuracy[i, indices[i, targets[i, j]]] += correct
            indices[i, targets[i, j]] += 1
    numpy_accuracy = np.mean(accuracy, axis=0) / 5
    theano_accuracy = accuracy_fn(predictions, targets)

    assert np.allclose(theano_accuracy, numpy_accuracy)
