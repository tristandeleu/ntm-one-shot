import theano
import theano.tensor as T
import numpy as np

import lasagne.nonlinearities

from .utils.init import weight_and_bias_init, shared_glorot_uniform, shared_one_hot
from .utils.similarities import cosine_similarity
from .utils.theano_utils import shared_floatX


def memory_augmented_neural_network(input_var, target_var, \
    batch_size=16, nb_class=5, memory_shape=(128, 40), \
    controller_size=200, input_size=20 * 20, nb_reads=4):
    """
    input_var has dimensions (batch_size, time, input_dim)
    target_var has dimensions (batch_size, time) (label indices)
    """

    M_0 = shared_floatX(1e-6 * np.ones((batch_size,) + memory_shape), name='memory')
    c_0 = shared_floatX(np.zeros((batch_size, controller_size)), name='memory_cell_state')
    h_0 = shared_floatX(np.zeros((batch_size, controller_size)), name='hidden_state')
    r_0 = shared_floatX(np.zeros((batch_size, nb_reads * memory_shape[1])), name='read_vector')
    wr_0 = shared_one_hot((batch_size, nb_reads, memory_shape[0]), name='wr')
    wu_0 = shared_one_hot((batch_size, memory_shape[0]), name='wu')

    W_key, b_key = weight_and_bias_init((controller_size, memory_shape[1]), name='key', n=nb_reads)
    W_add, b_add = weight_and_bias_init((controller_size, memory_shape[1]), name='add', n=nb_reads)
    W_sigma, b_sigma = weight_and_bias_init((controller_size, 1), name='sigma', n=nb_reads)
    # QKFIX: The scaling factor in Glorot initialisation is not correct if we
    # are computing the preactivations jointly
    W_xh, b_h = weight_and_bias_init((input_size + nb_class, 4 * controller_size), name='xh')
    W_rh = shared_glorot_uniform((nb_reads * memory_shape[1], 4 * controller_size), name='W_rh')
    W_hh = shared_glorot_uniform((controller_size, 4 * controller_size), name='W_hh')
    W_o, b_o = weight_and_bias_init((controller_size + nb_reads * memory_shape[1], nb_class), name='o')
    gamma = 0.95

    def slice_equally(x, size, nb_slices):
        return [x[:, n * size:(n + 1) * size] for n in range(nb_slices)]

    def step(x_t, M_tm1, c_tm1, h_tm1, r_tm1, wr_tm1, wu_tm1):
        # Feed Forward controller
        # h_t = lasagne.nonlinearities.tanh(T.dot(x_t, W_h) + b_h)
        # LSTM controller
        # p.3: "This memory is used by the controller as the input to a classifier,
        #       such as a softmax output layer, and as an additional
        #       input for the next controller state." -> T.dot(r_tm1, W_rh)
        preactivations = T.dot(x_t, W_xh) + T.dot(r_tm1, W_rh) + T.dot(h_tm1, W_hh) + b_h
        gf_, gi_, go_, u_ = slice_equally(preactivations, controller_size, 4)
        gf = lasagne.nonlinearities.sigmoid(gf_)
        gi = lasagne.nonlinearities.sigmoid(gi_)
        go = lasagne.nonlinearities.sigmoid(go_)
        u = lasagne.nonlinearities.tanh(u_)

        c_t = gf * c_tm1 + gi * u
        h_t = go * lasagne.nonlinearities.tanh(c_t) # (batch_size, num_units)

        k_t = lasagne.nonlinearities.tanh(T.dot(h_t, W_key) + b_key) # (batch_size, nb_reads, memory_size[1])
        a_t = lasagne.nonlinearities.tanh(T.dot(h_t, W_add) + b_add) # (batch_size, nb_reads, memory_size[1])
        sigma_t = lasagne.nonlinearities.sigmoid(T.dot(h_t, W_sigma) + b_sigma) # (batch_size, nb_reads, 1)
        sigma_t = T.addbroadcast(sigma_t, 2)

        wlu_tm1 = T.argsort(wu_tm1, axis=1)[:,:nb_reads] # (batch_size, nb_reads)
        # ww_t = sigma_t * wr_tm1 + (1. - sigma_t) * wlu_tm1
        ww_t = (sigma_t * wr_tm1).reshape((batch_size * nb_reads, memory_shape[0]))
        ww_t = T.inc_subtensor(ww_t[T.arange(batch_size * nb_reads), wlu_tm1.flatten()], 1. - sigma_t.flatten()) # (batch_size * nb_reads, memory_size[0])
        ww_t = ww_t.reshape((batch_size, nb_reads, memory_shape[0])) # (batch_size, nb_reads, memory_size[0])

        # p.4: "Prior to writing to memory, the least used memory location is
        #       computed from wu_tm1 and is set to zero"
        M_t = T.set_subtensor(M_tm1[T.arange(batch_size), wlu_tm1[:, 0]], 0.)
        M_t = M_t + T.batched_dot(ww_t.dimshuffle(0, 2, 1), a_t) # (batch_size, memory_size[0], memory_size[1])
        K_t = cosine_similarity(k_t, M_t) # (batch_size, nb_reads, memory_size[0])

        wr_t = lasagne.nonlinearities.softmax(K_t.reshape((batch_size * nb_reads, memory_shape[0])))
        wr_t = wr_t.reshape((batch_size, nb_reads, memory_shape[0])) # (batch_size, nb_reads, memory_size[0])
        if batch_size == 1:
            wr_t = T.unbroadcast(wr_t, 0)
        wu_t = gamma * wu_tm1 + T.sum(wr_t, axis=1) + T.sum(ww_t, axis=1) # (batch_size, memory_size[0])

        r_t = T.batched_dot(wr_t, M_t).flatten(ndim=2) # (batch_size, nb_reads * memory_size[1])

        return (M_t, c_t, h_t, r_t, wr_t, wu_t)

    # Model
    sequence_length_var = target_var.shape[1]
    output_shape_var = (batch_size * sequence_length_var, nb_class)

    # Join the input with time-offset labels
    one_hot_target_var_flatten = T.extra_ops.to_one_hot(target_var.flatten(), nb_class=nb_class)
    one_hot_target_var = one_hot_target_var_flatten.reshape((batch_size, sequence_length_var, nb_class))
    offset_target_var = T.concatenate([\
        T.zeros_like(one_hot_target_var[:,0]).dimshuffle(0, 'x', 1), \
        one_hot_target_var[:,:-1]], axis=1)
    l_input_var = T.concatenate([input_var, offset_target_var], axis=2)

    l_ntm_var, _ = theano.scan(step,
        sequences=[l_input_var.dimshuffle(1, 0, 2)],
        outputs_info=[M_0, c_0, h_0, r_0, wr_0, wu_0])
    l_ntm_output_var = T.concatenate(l_ntm_var[2:4], axis=2).dimshuffle(1, 0, 2)

    output_var_preactivation = T.dot(l_ntm_output_var, W_o) + b_o
    output_var_flatten = lasagne.nonlinearities.softmax(output_var_preactivation.reshape(output_shape_var))
    output_var = output_var_flatten.reshape(output_var_preactivation.shape)

    # Params
    params = [W_key, b_key, W_add, b_add, W_sigma, b_sigma, W_xh, W_rh, W_hh, b_h, W_o, b_o]

    return output_var, output_var_flatten, params