from __future__ import print_function

import theano
import theano.tensor as T
import numpy as np

import lasagne.updates
from mann.utils.generators import OmniglotGenerator
from mann.utils.metrics import accuracy_instance
from mann.model import memory_augmented_neural_network
import time


def omniglot():
    input_var = T.tensor3('input') # input_var has dimensions (batch_size, time, input_dim)
    target_var = T.imatrix('target') # target_var has dimensions (batch_size, time) (label indices)

    # Load data
    generator = OmniglotGenerator(data_folder='./data/omniglot', batch_size=16, \
        nb_samples=5, nb_samples_per_class=10, max_rotation=0., max_shift=0, max_iter=None)

    output_var, output_var_flatten, params = memory_augmented_neural_network(input_var, \
        target_var, batch_size=generator.batch_size, nb_class=generator.nb_samples, \
        memory_shape=(128, 40), controller_size=200, input_size=20 * 20, nb_reads=4)

    cost = T.mean(T.nnet.categorical_crossentropy(output_var_flatten, target_var.flatten()))
    updates = lasagne.updates.adam(cost, params, learning_rate=1e-3)

    accuracies = accuracy_instance(T.argmax(output_var, axis=2), target_var, batch_size=generator.batch_size)

    print('Compiling the model...')
    train_fn = theano.function([input_var, target_var], cost, updates=updates)
    accuracy_fn = theano.function([input_var, target_var], accuracies)
    print('Done')

    print('Training...')
    t0 = time.time()
    all_scores, scores, accs = [], [], np.zeros(generator.nb_samples_per_class)
    try:
        for i, (example_input, example_output) in generator:
            score = train_fn(example_input, example_output)
            acc = accuracy_fn(example_input, example_output)
            all_scores.append(score)
            scores.append(score)
            accs += acc
            if i > 0 and not (i % 100):
                print('Episode %05d: %.6f' % (i, np.mean(score)))
                print(accs / 100.)
                scores, accs = [], np.zeros(generator.nb_samples_per_class)
    except KeyboardInterrupt:
        print(time.time() - t0)
        pass


if __name__ == '__main__':
    omniglot()
