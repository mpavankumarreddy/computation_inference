import os
import random

import tensorflow as tf
import numpy as np
from keras.utils.generic_utils import Progbar
import matplotlib.pyplot as plt
import mnist
import string
import cPickle as pickle


def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


class SemiSup(object):

    def __init__(self, data_dim, latent_dim, hidden_dims, num_classes):
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        # array of hidden dims used both for inference network and decoder network
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.x = tf.placeholder(tf.float32, shape=(None, data_dim), name="input")
        self.y_gold = tf.placeholder(tf.float32, shape=(None, num_classes), name="gold_labels")
        self.prior_class_probs_params = \
            tf.nn.softmax(tf.Variable([[1. / num_classes] * num_classes], trainable=True, name="prior_class_probs_params"))
        # since we have to loop over y, the likelihood lower bound L(x, y) should be generated for different y
        self.batch_size = tf.shape(self.x)[0]
        self.eta = tf.random_normal((self.batch_size, self.latent_dim), mean=0., stddev=1.)

        # assuming one-hot input for y
        # self.y = tf.placeholder(tf.float32, shape=(None, num_classes), name="label")
        # prior probability parameters of classes
        self.inference_network_reuse_params = None
        self.decoder_network_reuse_params = None
        elbos = []
        z_mu_infs, z_log_sigma_sq_infs, z_sampled_arr = [], [], []
        x_mu_decs, x_log_sigma_sq_decs = [], []
        for i in xrange(num_classes):
            y_index = i
            z_mu_inf, z_log_sigma_sq_inf, z_sampled, x_mu_dec, x_log_sigma_sq_dec, elbo_for_y = \
                self.get_calculations_for_y(y_index)
            z_mu_infs.append(z_mu_inf)
            z_log_sigma_sq_infs.append(z_log_sigma_sq_inf)
            z_sampled_arr.append(z_sampled)
            x_mu_decs.append(x_mu_dec)
            x_log_sigma_sq_decs.append(x_log_sigma_sq_dec)
            elbos.append(elbo_for_y)

        self.z_mu_infs = z_mu_infs
        self.z_log_sigma_sq_infs = z_log_sigma_sq_infs
        self.z_sampled_arr = z_sampled_arr
        self.x_mu_decs = x_mu_decs
        self.x_log_sigma_sq_decs = x_log_sigma_sq_decs

        # elbos doesn't include log p(y). Add the correction
        self.elbos = tf.pack(elbos, axis=1) + tf.log(1e-10 + tf.reshape(self.prior_class_probs_params, shape=(1, -1)))
        self.qy_x = \
            self.get_multinomial_mlp(data_dim, hidden_dims, num_classes, self.x, scope_name="qy_x", reuse_params=None)
        # entropy of qy_x
        H_qy_x = -tf.reduce_sum(self.qy_x * tf.log(1e-10 + self.qy_x), reduction_indices=1)
        # supervised loss
        self.supervised_generative_loss = -tf.reduce_mean(tf.reduce_sum(self.elbos * self.y_gold, reduction_indices=1))
        self.supervised_discriminative_loss = \
            -tf.reduce_mean(tf.reduce_sum(self.y_gold * tf.log(1e-10 + self.qy_x), reduction_indices=1))
        self.supervised_loss = self.supervised_generative_loss + .1*self.supervised_discriminative_loss
        self.unsupervised_loss = \
            -tf.reduce_mean(tf.reduce_sum(self.elbos * self.qy_x, reduction_indices=1) + H_qy_x)
        optimizer = tf.train.AdamOptimizer()
        sup_gradients = optimizer.compute_gradients(self.supervised_loss)
        capped_sup_grads = [(tf.clip_by_value(grad, -100., 100.) if grad is not None else None, var) for grad, var in sup_gradients]
        self.supervised_train_op = optimizer.apply_gradients(capped_sup_grads)
        self.unsupervised_train_op = optimizer.minimize(self.unsupervised_loss)

    def get_calculations_for_y(self, y_index):
        y_onehot = [0.] * num_classes
        y_onehot[y_index] = 1.
        y = tf.tile(tf.constant([y_onehot]), [self.batch_size, 1])
        z_mu_inf, z_log_sigma_sq_inf = \
            self.get_mean_sigma_mlp_with_two_inputs(
                self.data_dim, self.num_classes, self.hidden_dims, self.latent_dim, self.x, y, "qz_xy", reuse_params=self.inference_network_reuse_params)
        self.inference_network_reuse_params = True
        z_sampled = tf.add(z_mu_inf, tf.sqrt(tf.exp(z_log_sigma_sq_inf)) * self.eta, name="z_sampled_%d" % y_index)
        x_mu_dec, x_log_sigma_sq_dec = \
            self.get_mean_sigma_mlp_with_two_inputs(
                self.latent_dim, self.num_classes, self.hidden_dims, self.data_dim, z_sampled, y, "decoder_network", reuse_params=self.decoder_network_reuse_params)
        self.decoder_network_reuse_params = True
        # greyscale images
        x_mu_dec = tf.nn.sigmoid(x_mu_dec, name="x_mu_dec_%d" % y_index)
        elbo_for_y = self.get_elbo_for_y(y, x_mu_dec, x_log_sigma_sq_dec, z_mu_inf, z_log_sigma_sq_inf)
        return z_mu_inf, z_log_sigma_sq_inf, z_sampled, x_mu_dec, x_log_sigma_sq_dec, elbo_for_y

    def get_elbo_for_y(self, y, x_mu_dec, x_log_sigma_sq_dec, z_mu_inf, z_log_sigma_sq_inf):
        # -E[log p(x|z)]
        reconstr_loss = -tf.reduce_sum(self.x * tf.log(1e-10 + x_mu_dec) +
                                       (1 - self.x) * tf.log(1e-10 + 1 - x_mu_dec),
                                       reduction_indices=1)
        kl_div = \
            -0.5 * tf.reduce_sum(1 + z_log_sigma_sq_inf
                                - tf.exp(z_log_sigma_sq_inf)
                                - tf.square(z_mu_inf), reduction_indices=1)
        elbo = - reconstr_loss - kl_div
        return elbo

    def get_mean_sigma_mlp_with_two_inputs(self, input_dim1, input_dim2, hidden_dims, output_dim, input_x1, input_x2,
                                           scope_name, reuse_params):
        with tf.variable_scope(scope_name, reuse=reuse_params):
            w_i1_h1 = tf.get_variable(name="input1_hidden1_weights", shape=(input_dim1, hidden_dims[0]),
                                      initializer=tf.contrib.layers.xavier_initializer())
            b_i1_h1 = tf.get_variable(initializer=tf.zeros_initializer((hidden_dims[0],)),
                                      name="input1_hidden1_biases")
            h1_x1 = tf.nn.relu(tf.matmul(input_x1, w_i1_h1) + b_i1_h1)

            w_i2_h1 = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                                      shape=(input_dim2, hidden_dims[0]), name="input2_hidden1_weights")
            b_i2_h1 = tf.get_variable(initializer=tf.zeros_initializer((hidden_dims[0],)),
                                      name="input2_hidden1_biases")
            h1_x2 = tf.nn.relu(tf.matmul(input_x2, w_i2_h1) + b_i2_h1)

            h = h1_x1 + h1_x2

            for i in xrange(1, len(hidden_dims[1:])+1):
                j = i + 1
                hidden_dim_i = hidden_dims[i-1]
                hidden_dim_j = hidden_dims[j-1]

                w_hi_hj = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                                          shape=(hidden_dim_i, hidden_dim_j), name="hidden%d_hidden%d_weights" % (i, j))
                b_hi_hj = tf.get_variable(initializer=tf.zeros_initializer((hidden_dim_j,)),
                                          name="hidden%d_hidden%d_biases" % (i, j))
                h = tf.nn.relu(tf.matmul(h, w_hi_hj) + b_hi_hj)

            hidden_dim = hidden_dims[-1]
            w_mean = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                                     shape=(hidden_dim, output_dim), name="mean_weights")
            b_mean = tf.get_variable(initializer=tf.zeros_initializer((output_dim,)),
                                     name="mean_biases")
            w_log_sigma_sq = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                                             shape=(hidden_dim, output_dim), name="log_sigma_sq_weights")
            b_log_sigma_sq = tf.get_variable(initializer=tf.zeros_initializer((output_dim,)),
                                             name="log_sigma_sq_biases")

            mean = tf.matmul(h, w_mean) + b_mean
            log_sigma_sq = tf.matmul(h, w_log_sigma_sq) + b_log_sigma_sq
        return mean, log_sigma_sq

    def get_multinomial_mlp(self, input_dim, hidden_dims, output_dim, input_x, scope_name, reuse_params):
        with tf.variable_scope(scope_name, reuse=reuse_params):
            w_h1 = tf.get_variable(name="input_hidden1_weights", shape=(input_dim, hidden_dims[0]),
                                      initializer=tf.contrib.layers.xavier_initializer())
            b_h1 = tf.get_variable(initializer=tf.zeros_initializer((hidden_dims[0],)),
                                      name="input_hidden1_biases")
            h = tf.nn.relu(tf.matmul(input_x, w_h1) + b_h1)

            for i in xrange(1, len(hidden_dims[1:]) + 1):
                j = i + 1
                hidden_dim_i = hidden_dims[i - 1]
                hidden_dim_j = hidden_dims[j - 1]

                w_hi_hj = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                                          shape=(hidden_dim_i, hidden_dim_j), name="hidden%d_hidden%d_weights" % (i, j))
                b_hi_hj = tf.get_variable(initializer=tf.zeros_initializer((hidden_dim_j,)),
                                          name="hidden%d_hidden%d_biases" % (i, j))
                h = tf.nn.relu(tf.matmul(h, w_hi_hj) + b_hi_hj)

            hidden_dim = hidden_dims[-1]
            w_softmax = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                                     shape=(hidden_dim, output_dim), name="softmax_weights")
            b_softmax = tf.get_variable(initializer=tf.zeros_initializer((output_dim,)),
                                     name="softmax_biases")
            output = tf.nn.softmax(tf.matmul(h, w_softmax) + b_softmax)
        return output

    def decode(self, sess, z_mu=None, y=None):
        """ Generate data by sampling from latent space.

        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent
        space. Same for y
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.latent_dim)
        if y is None:
            y_index = np.random.randint(self.num_classes)
        else:
            y_index = np.argmax(y)
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return sess.run(self.x_mu_decs[y_index], feed_dict={self.z_sampled_arr[y_index]: [z_mu], self.batch_size: 1})

    def reconstruct(self, sess, X):
        """ Use VAE to reconstruct given data. """
        # Get the most likely z from the posterior distribution
        z_mu_inf = self.transform(sess, X)
        # Reconstruct from the most likely z
        return sess.run(self.x_mu_dec, feed_dict={self.z_sampled: z_mu_inf})

    def transform_with_labels(self, sess, X, labels):
        """ Use VAE to reconstruct given data. """
        # Get the most likely z from the posterior distribution
        z_mu_infs = sess.run(self.z_mu_infs, feed_dict={self.x: X})
        z_mu_inf = [items[label] for items, label in zip(z_mu_infs, labels)]
        return np.array(z_mu_inf)

    def make_batches(self, size, batch_size):
        '''Returns a list of batch indices (tuples of indices).
        '''
        nb_batch = int(np.ceil(size / float(batch_size)))
        return [(i * batch_size, min(size, (i + 1) * batch_size))
                for i in range(0, nb_batch)]

    def run_train_epoch(self, session, x_l, y_l, x_u, num_batches, shuffle=True, verbose=1):
        n_l_samples = len(x_l)
        no_unsup = x_u is None
        l_batch_size = n_l_samples / num_batches
        l_index_array = np.arange(l_batch_size * num_batches)
        if shuffle:
            np.random.shuffle(l_index_array)
        l_batches = self.make_batches(l_batch_size * num_batches, l_batch_size)
        if not no_unsup:
            n_u_samples = len(x_u)
            u_batch_size = n_u_samples / num_batches
            u_index_array = np.arange(u_batch_size * num_batches)
            if shuffle: np.random.shuffle(u_index_array)
            u_batches = self.make_batches(u_batch_size * num_batches, u_batch_size)
        progbar = Progbar(num_batches)
        avg_total_l_loss = 0.
        avg_total_u_loss = 0.
        total_l_samples = 0.
        total_u_samples = 0.
        for batch_index in xrange(num_batches):
            l_batch_start, l_batch_end = l_batches[batch_index]
            l_batch_ids = l_index_array[l_batch_start:l_batch_end]
            x_l_batch = x_l[l_batch_ids]
            y_l_batch = y_l[l_batch_ids]
            _, l_loss = \
                session.run([self.supervised_train_op, self.supervised_loss],
                            {self.x: x_l_batch, self.y_gold: y_l_batch})
            if np.isnan(l_loss) or np.isinf(l_loss):
                raise ValueError("nan or inf l_loss")
            total_l_samples += l_batch_size
            avg_total_l_loss += (l_loss * l_batch_size / n_l_samples)
            if not no_unsup:
                u_batch_start, u_batch_end = u_batches[batch_index]
                u_batch_ids = u_index_array[u_batch_start:u_batch_end]
                x_u_batch = x_u[u_batch_ids]
                _, u_loss = \
                    session.run([self.unsupervised_train_op, self.unsupervised_loss],
                                {self.x: x_u_batch})
                if np.isnan(u_loss) or np.isinf(u_loss):
                    raise ValueError("nan or inf u_loss")
                total_u_samples += u_batch_size
                avg_total_u_loss += (u_loss * u_batch_size / n_u_samples)
            if verbose == 1:
                progbar.update(
                    batch_index+1,
                    values=[("avg_l_loss/1000", avg_total_l_loss * 1000. / total_l_samples),
                            ("avg_u_loss/1000", 0.)
                            if no_unsup else ("avg_u_loss/1000", avg_total_u_loss * 1000. / total_u_samples)],
                    force=True)
        print("avg total l_loss, u_loss = %.2f, %.2f" % (avg_total_l_loss, avg_total_u_loss))


def plot_canvas(sess, model_2d, digit, canvas_range=3, n_labels=10, imdim=28, cmap_name='Greys'):
    assert model_2d.latent_dim == 2
    nx = ny = 20
    x_values = np.linspace(-canvas_range, canvas_range, nx)
    y_values = np.linspace(-canvas_range, canvas_range, ny)
    label_onehot = np.zeros((n_labels,))
    label_onehot[digit] = 1

    canvas = np.empty((imdim * ny, imdim * nx))
    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            z_mu = np.array([xi, yi])
            x_mean = model_2d.decode(sess, z_mu, label_onehot)
            canvas[(nx - i - 1) * imdim:(nx - i) * imdim, j * imdim:(j + 1) * imdim] = x_mean[0].reshape(imdim, imdim)

    plt.figure(figsize=(8, 10))
    # Xi, Yi = np.meshgrid(x_values, y_values)
    plt.imshow(canvas, cmap=plt.get_cmap(cmap_name))
    plt.tight_layout()


def plot_style(sess, model, x_test, y_index_test, n, n_labels=10, imdim=28, shuffle=False, y_indices_to_plot=None):
    plt.figure(figsize=(4*n, 30))
    # indices = np.arange(n) if not shuffle else np.random.randint(len(x_test), size=n)
    indices = np.arange(n) if not shuffle else np.random.choice(xrange(200), size=n, replace=False)
    y_indices_to_plot = range(n_labels) if y_indices_to_plot is None else y_indices_to_plot
    for i, index in enumerate(indices):
        # display original
        ax = plt.subplot(1+len(y_indices_to_plot), n, i + 1)
        plt.subplots_adjust(hspace=.01)
        plt.imshow(x_test[index].reshape(imdim, imdim))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        z_mu = sess.run(model.z_mu_infs[y_index_test[index]], feed_dict={model.x: [x_test[index]]})[0]

        for row, j in enumerate(y_indices_to_plot):

            # display reconstruction
            ax = plt.subplot(1+len(y_indices_to_plot), n, i + 1 + n*(row+1))
            plt.subplots_adjust(hspace=.01)
            label_onehot = np.zeros((n_labels,))
            label_onehot[j] = 1
            plt.imshow(model.decode(sess, z_mu, label_onehot).reshape(imdim, imdim))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    # plt.tight_layout()


# good labels to plot for fonts
good_fnt_labels = ['0', 'B', 'C', 'E', 'S', 'G', 'P']

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Semi-supervised Deep Generative Models')
    parser.add_argument('-n_labeled', metavar='n_labeled', type=int, required=True,
                        help='number of labeled examples to be used.')
    parser.add_argument('-h1', metavar='hidden1_dim', type=int, default=50,
                        help='number of nodes in 1st hidden layer.')
    parser.add_argument('-h2', metavar='hidden2_dim', type=int, default=50,
                        help='number of nodes in 2nd hidden layer.')
    parser.add_argument('-latent_dim', type=int, default=50,
                        help='number of latent variables to be used.')
    parser.add_argument('-nb_epochs', type=int, default=100,
                        help='number of epochs to train.')
    parser.add_argument('-num_batches', type=int, default=10,
                        help='number of batches per epoch.')
    parser.add_argument('--save', dest='save', action='store_true', help='whether to save the model')
    parser.add_argument('--no-save', dest='save', action='store_false', help='whether to save the model')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true', help='whether to overwrite a saved model')
    parser.add_argument('--no-overwrite', dest='overwrite', action='store_false', help='whether to overwrite a saved model')
    parser.add_argument('--fontdata', dest='fontdata', action='store_true', help='mnist or fonts dataset')
    parser.add_argument('--no-fontdata', dest='fontdata', action='store_false', help='mnist or fonts dataset')
    args = parser.parse_args()

    np.random.seed(123)
    random.seed(123)
    size = 28  # only 28 dataset is available
    n_labeled = args.n_labeled
    if args.fontdata:
        with open("../fnts_data.pkl") as fp:
            fnts_data = pickle.load(fp)
        chars = map(str, range(10)) + list(string.ascii_uppercase)
        char2index = {char: index for index, char in enumerate(chars)}
        train_x = []
        train_y = []
        for char in chars:
            char_data = fnts_data[char]
            char_onehot = np.zeros((len(chars),))
            char_onehot[char2index[char]] = 1.
            for datum in char_data:
                train_x.append(np.reshape(datum, (-1,)))
                train_y.append(char_onehot)
        train_x = 1. - np.array(train_x)/255.
        train_y = np.array(train_y)
        x_l, y_l = train_x, train_y
        x_u, y_u = None, None
    else:
        train_x, train_y, valid_x, valid_y, test_x, test_y = mnist.load_numpy_split(size, binarize_y=True)
        # transpose all array from the library
        x_l, y_l, x_u, y_u = [arr.T for arr in mnist.create_semisupervised(train_x, train_y, n_labeled)]


    data_dim = x_l.shape[1]
    hidden_dims = [args.h1, args.h2]
    latent_dim = args.latent_dim
    num_classes = y_l.shape[1]

    model = SemiSup(data_dim, latent_dim, hidden_dims, num_classes)
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    sess = tf.Session(config=config_proto)
    sess.run(tf.initialize_all_variables())
    model_path = "./save/ss_dgm/h1_%d_h2_%d_latent_%d_nl_%d" % (args.h1, args.h2, args.latent_dim, args.n_labeled)
    model_path += '_fnts' if args.fontdata else ''
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    ckpt = tf.train.get_checkpoint_state(model_path)
    found_saved = False
    if ckpt and ckpt.model_checkpoint_path:
        print("Found a saved model. Restoring the model")
        tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)
        found_saved = True
    else:
        print("Didn't find any saved model.")
    if not found_saved or args.overwrite:
        summary_writer = tf.train.SummaryWriter('./logs/ss_dgm_log', sess.graph)
        print(("Resuming" if found_saved else "Starting") + " training")
        try:
            for epoch in xrange(1, args.nb_epochs+1):
                print("Epoch [%d]" % epoch)
                model.run_train_epoch(sess, x_l, y_l, x_u, args.num_batches, shuffle=True, verbose=1)
                if epoch % 2 == 0:
                    tf.train.Saver().save(sess, model_path + '/model.ckpt')
        except KeyboardInterrupt:
            print("\ntraining interrupted.")
        tf.train.Saver().save(sess, model_path + '/model.ckpt')

