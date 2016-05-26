import theano
import theano.tensor as T
import numpy as np
import os
import random

import lasagne.nonlinearities
import lasagne.updates
from utils.init import weight_and_bias_init, shared_glorot_uniform, shared_one_hot
from utils.similarities import cosine_similarity
from utils.theano_utils import shared_floatX
from utils.images import get_shuffled_images, time_offset_input
from utils.generators import OmniglotGenerator
from utils.metrics import accuracy_instance
import time

nb_class = 5
memory_shape = (128, 40)
controller_size = 200
input_size = 20 * 20 + nb_class

floatX = theano.config.floatX

# TODO: Fill in the initial parameters
M_0 = shared_floatX(1e-6 * np.ones(memory_shape), name='memory')
c_0 = shared_floatX(np.zeros(controller_size), name='memory_cell_state')
h_0 = shared_floatX(np.zeros(controller_size), name='hidden_state')
r_0 = shared_floatX(np.zeros(memory_shape[1]), name='read_vector')
wr_0 = shared_one_hot(memory_shape[0], name='wr')
ww_0 = shared_one_hot(memory_shape[0], name='ww')
wu_0 = shared_one_hot(memory_shape[0], name='wu')

# TODO: Fill the weights
W_key, b_key = weight_and_bias_init((controller_size, memory_shape[1]), name='key')
W_add, b_add = weight_and_bias_init((controller_size, memory_shape[1]), name='add')
W_sigma, b_sigma = weight_and_bias_init((controller_size, 1), name='sigma')
# QKFIX: The scaling factor in Glorot initialisation is not correct if we
# are computing the preactivations jointly
W_xh, b_h = weight_and_bias_init((input_size, 4 * controller_size), name='xh')
# QKFIX: Only 1 read head
W_rh = shared_glorot_uniform((memory_shape[1], 4 * controller_size), name='W_rh')
W_hh = shared_glorot_uniform((controller_size, 4 * controller_size), name='W_hh')
W_o, b_o = weight_and_bias_init((controller_size + memory_shape[1], nb_class), name='o')
gamma = 0.95

def slice_preactivations(x):
    return [x[n*controller_size:(n+1)*controller_size] for n in range(4)]

def step(x_t, M_tm1, c_tm1, h_tm1, r_tm1, wr_tm1, ww_tm1, wu_tm1):
    # Feed Forward controller
    # h_t = lasagne.nonlinearities.tanh(T.dot(x_t, W_h) + b_h)
    # LSTM controller
    # p.3: "This memory is used by the controller as the input to a classifier,
    #       such as a softmax output layer, and as an additional
    #       input for the next controller state."
    preactivations = T.dot(x_t, W_xh) + T.dot(r_tm1, W_rh) + T.dot(h_tm1, W_hh) + b_h
    gf_, gi_, go_, u_ = slice_preactivations(preactivations)
    gf = lasagne.nonlinearities.sigmoid(gf_)
    gi = lasagne.nonlinearities.sigmoid(gi_)
    go = lasagne.nonlinearities.sigmoid(go_)
    u = lasagne.nonlinearities.tanh(u_)

    c_t = gf * c_tm1 + gi * u
    h_t = go * lasagne.nonlinearities.tanh(c_t)

    k_t = lasagne.nonlinearities.tanh(T.dot(h_t, W_key) + b_key)
    a_t = lasagne.nonlinearities.tanh(T.dot(h_t, W_add) + b_add)
    sigma_t = lasagne.nonlinearities.sigmoid(T.dot(h_t, W_sigma) + b_sigma)

    # "n is set to be the number of reads to memory" -> n is the number of read heads
    # QKFIX: set n = 1 for now for a single read head
    # TODO: case where there are multiple read heads
    wlu_tm1 = T.argmin(wu_tm1, axis=0)
    # ww_t = sigma_t * wr_tm1 + (1. - sigma_t) * wlu_tm1
    ww_t = sigma_t[0] * wr_tm1
    ww_t = T.inc_subtensor(ww_t[wlu_tm1], 1. - sigma_t[0])

    M_t = M_tm1 + T.outer(ww_t, a_t)
    K_t = cosine_similarity(k_t, M_t)

    # softmax returns a row vector with shape (1, memory_size[0])
    wr_t = lasagne.nonlinearities.softmax(K_t)[0]
    wu_t = gamma * wu_tm1 + wr_t + ww_t

    r_t = T.dot(wr_t, M_t)

    return (M_t, c_t, h_t, r_t, wr_t, ww_t, wu_t)

##
# Model
##
input_var = T.matrix('input') # input_var has dimensions (time, input_dim)
target_var = T.ivector('target') # target_var has dimensions (time,) (label indices)

# Join the input with time-offset labels
one_hot_target_var = T.extra_ops.to_one_hot(target_var, nb_class=nb_class)
offset_target_var = T.concatenate([\
    T.zeros_like(one_hot_target_var[0]).dimshuffle('x', 0), \
    one_hot_target_var[:-1]], axis=0)
l_input_var = T.concatenate([input_var, offset_target_var], axis=1)

l_ntm_var, _ = theano.scan(step,
    sequences=[l_input_var],
    outputs_info=[M_0, c_0, h_0, r_0, wr_0, ww_0, wu_0])
l_ntm_output_var = T.concatenate(l_ntm_var[2:4], axis=1)

# TODO: add dense layer on top + softmax activation
output_var = lasagne.nonlinearities.softmax(T.dot(l_ntm_output_var, W_o) + b_o)

cost = T.mean(T.nnet.categorical_crossentropy(output_var, target_var))
params = [W_key, b_key, W_add, b_add, W_sigma, b_sigma, W_xh, W_rh, W_hh, b_h, W_o, b_o]
updates = lasagne.updates.adam(cost, params, learning_rate=1e-4)

accuracies = accuracy_instance(T.argmax(output_var, axis=1), target_var)

train_fn = theano.function([input_var, target_var], cost, updates=updates)
accuracy_fn = theano.function([input_var, target_var], accuracies)

##
# Load data
##
generator = OmniglotGenerator(data_folder='./data/omniglot', nb_samples=5, \
    nb_samples_per_class=10, max_rotation=0., max_shift=0, max_iter=None)
t0 = time.time()
all_scores, scores, accs = [], [], np.zeros(generator.nb_samples_per_class)
try:
    for i, (example_input, example_output) in generator:
        score = train_fn(example_input, example_output)
        acc = accuracy_fn(example_input, example_output)
        all_scores.append(score)
        scores.append(score)
        accs += acc
        if i > 0 and not (i % 1000):
            print 'Episode %05d: %.6f' % (i, np.mean(score))
            print accs / 1000.
            scores, accs = [], np.zeros(generator.nb_samples_per_class)
except KeyboardInterrupt:
    print time.time() - t0
    pass