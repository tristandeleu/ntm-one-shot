import pytest

import theano
import theano.tensor as T
import numpy as np

from .model import memory_augmented_neural_network

def test_batch_size():
    input_var_1, input_var_2 = T.tensor3s('input1', 'input2')
    target_var_1, target_var_2 = T.imatrices('target1', 'target2')
    # First model with `batch_size=16`
    output_var_1, _, params1 = memory_augmented_neural_network(
        input_var_1, target_var_1,
        batch_size=16,
        nb_class=5,
        memory_shape=(128, 40),
        controller_size=200,
        input_size=20 * 20,
        nb_reads=4)
    # Second model with `batch_size=1`
    output_var_2, _, params2 = memory_augmented_neural_network(
        input_var_2, target_var_2,
        batch_size=1,
        nb_class=5,
        memory_shape=(128, 40),
        controller_size=200,
        input_size=20 * 20,
        nb_reads=4)

    for (param1, param2) in zip(params1, params2):
        param2.set_value(param1.get_value())

    posterior_fn1 = theano.function([input_var_1, target_var_1], output_var_1)
    posterior_fn2 = theano.function([input_var_2, target_var_2], output_var_2)

    # Input has shape (batch_size, timesteps, vocabulary_size + actions_vocabulary_size + 3)
    test_input = np.random.rand(16, 50, 20 * 20)
    test_target = np.random.randint(5, size=(16, 50)).astype('int32')

    test_output1 = posterior_fn1(test_input, test_target)
    test_output2 = np.zeros_like(test_output1)

    for i in range(16):
        test_output2[i] = posterior_fn2(test_input[i][np.newaxis, :, :], test_target[i][np.newaxis, :])

    assert np.allclose(test_output1, test_output2)

def test_shape():
    input_var = T.tensor3('input')
    target_var = T.imatrix('target')
    output_var, _, _ = memory_augmented_neural_network(
        input_var, target_var,
        batch_size=16,
        nb_class=5,
        memory_shape=(128, 40),
        controller_size=200,
        input_size=20 * 20,
        nb_reads=4)

    posterior_fn = theano.function([input_var, target_var], output_var)

    test_input = np.random.rand(16, 50, 20 * 20)
    test_target = np.random.randint(5, size=(16, 50)).astype('int32')
    test_input_invalid_batch_size = np.random.rand(16 + 1, 50, 20 * 20)
    test_input_invalid_depth = np.random.rand(16, 50, 20 * 20 - 1)
    test_output = posterior_fn(test_input, test_target)

    assert test_output.shape == (16, 50, 5)
    with pytest.raises(ValueError) as e_info:
        posterior_fn(test_input_invalid_batch_size, test_target)
    with pytest.raises(ValueError) as e_info:
        posterior_fn(test_input_invalid_depth, test_target)
