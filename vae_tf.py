import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from keras.utils.generic_utils import Progbar
import matplotlib.pyplot as plt


def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


class VAE(object):

    def __init__(self, data_dim, latent_dim, hidden_dim):
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.x = tf.placeholder(tf.float32, shape=(None, data_dim), name="input")
        self.batch_size = tf.shape(self.x)[0]
        eta = tf.random_normal((self.batch_size, self.latent_dim), mean=0., stddev=1.)
        self.z_mu_inf, self.z_log_sigma_sq_inf = \
            self.get_mlp_network_for_mean_logsigma_sq(data_dim, hidden_dim, latent_dim, self.x, "inference_network")
        self.z_sampled = self.z_mu_inf + tf.sqrt(tf.exp(self.z_log_sigma_sq_inf)) * eta
        self.x_mu_dec, self.x_log_sigma_sq_dec = \
            self.get_mlp_network_for_mean_logsigma_sq(latent_dim, hidden_dim, data_dim, self.z_sampled, "decoder_network")
        self.x_mu_dec = tf.nn.sigmoid(self.x_mu_dec)
        self.build_loss_term()

    def get_mlp_network_for_mean_logsigma_sq(self, input_dim, hidden_dim, output_dim, input_x, scope_name):
        with tf.variable_scope(scope_name):
            w_h = tf.Variable(xavier_init(input_dim, hidden_dim),
                              name="hidden_weights")
            b_h = tf.Variable(tf.zeros((hidden_dim,)), name="hidden_biases")

            w_mean = tf.Variable(xavier_init(hidden_dim, output_dim),
                                 name="mean_weights")
            b_mean = tf.Variable(tf.zeros((output_dim,)), name="mean_biases")

            w_log_sigma_sq = tf.Variable(xavier_init(hidden_dim, output_dim),
                                         name="log_sigma_sq_weights")
            b_log_sigma_sq = tf.Variable(tf.zeros((output_dim,)), name="log_sigma_sq_biases")

            h = tf.nn.relu(tf.matmul(input_x, w_h) + b_h)
            mean = tf.matmul(h, w_mean) #+ b_mean
            log_sigma_sq = tf.matmul(h, w_log_sigma_sq) #+ b_log_sigma_sq

            return mean, log_sigma_sq

    def generate(self, sess, z_mu=None):
        """ Generate data by sampling from latent space.

        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent
        space.
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.latent_dim)
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return sess.run(self.x_mu_dec, feed_dict={self.z_sampled: z_mu})

    def reconstruct(self, sess, X):
        """ Use VAE to reconstruct given data. """
        # Get the most likely z from the posterior distribution
        z_mu_inf = self.transform(sess, X)
        # Reconstruct from the most likely z
        return sess.run(self.x_mu_dec, feed_dict={self.z_sampled: z_mu_inf})

    def transform(self, sess, X):
        """ Use VAE to reconstruct given data. """
        # Get the most likely z from the posterior distribution
        z_mu_inf = sess.run(self.z_mu_inf, feed_dict={self.x: X})
        return z_mu_inf

    def build_loss_term(self):
        D = self.data_dim
        # self.reconstr_loss = \
        #     -((-D/2.) * np.log(2*np.pi) - .5 * tf.reduce_sum(self.x_log_sigma_sq_dec, reduction_indices=1) - 0.5 * (
        #     tf.reduce_sum(tf.square(self.x - self.x_mu_dec) / tf.exp(self.x_log_sigma_sq_dec), reduction_indices=1)))

        self.reconstr_loss = -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_mu_dec) +
                                            (1 - self.x) * tf.log(1e-10 + 1 - self.x_mu_dec),
                                            reduction_indices=1)
        # self.reconstr_loss = \
        #     tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(self.x_mu_dec, self.x), reduction_indices=1)
        self.kl_div_loss = \
            -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq_inf
                                - tf.exp(self.z_log_sigma_sq_inf)
                                - tf.square(self.z_mu_inf), reduction_indices=1)
        self.loss = tf.reduce_mean(self.reconstr_loss + self.kl_div_loss)
        optimizer = tf.train.AdamOptimizer()
        # grads = optimizer.compute_gradients(self.loss, tf.trainable_variables())
        # capped_grads = [( tf.clip_by_value(grad, -5., 5.) if grad is not None else None, var ) for grad, var in grads]
        # self.train_op = optimizer.apply_gradients(capped_grads)
        self.train_op = optimizer.minimize(self.loss)

    def make_batches(self, size, batch_size):
        '''Returns a list of batch indices (tuples of indices).
        '''
        nb_batch = int(np.ceil(size / float(batch_size)))
        return [(i * batch_size, min(size, (i + 1) * batch_size))
                for i in range(0, nb_batch)]

    def run_train_epoch(self, session, x_inputs, batch_size, shuffle=True, verbose=1):
        num_samples = len(x_inputs)
        index_array = np.arange(num_samples)
        if shuffle:
            np.random.shuffle(index_array)
        batches = self.make_batches(num_samples, batch_size)
        nb_batch = len(batches)
        progbar = Progbar(nb_batch)
        avg_total_loss = 0.
        total_samples = 0.
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            x_batch = x_inputs[batch_ids]
            _, loss = session.run([self.train_op, self.loss], {self.x: x_batch})
            if np.isnan(loss) or np.isinf(loss):
                raise ValueError("nan or inf loss")
            cur_batch_size = (batch_end - batch_start)
            total_samples += cur_batch_size
            avg_total_loss += (loss * cur_batch_size / num_samples)
            if verbose == 1:
                progbar.update(
                    batch_index+1,
                    values=[("avg loss per 1000 samples", avg_total_loss * 1000. / total_samples)], force=True)
        print("avg total loss = %d" % avg_total_loss)


def plot_canvas(sess, vae_2d):
    assert vae_2d.latent_dim == 2
    nx = ny = 20
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)

    canvas = np.empty((28 * ny, 28 * nx))
    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            z_mu = np.array([[xi, yi]])
            x_mean = vae_2d.generate(sess, z_mu)
            canvas[(nx - i - 1) * 28:(nx - i) * 28, j * 28:(j + 1) * 28] = x_mean[0].reshape(28, 28)

    plt.figure(figsize=(8, 10))
    # Xi, Yi = np.meshgrid(x_values, y_values)
    plt.imshow(canvas)
    plt.tight_layout()

if __name__ == '__main__':
    (x_train, y_test), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    data_dim = x_train.shape[1]
    hidden_dim = 512
    latent_dim = 2
    vae = VAE(data_dim, latent_dim, hidden_dim)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    nb_epochs = 100
    try:
        for epoch in range(nb_epochs):
            print("Epoch [%d]" % (epoch+1))
            vae.run_train_epoch(sess, x_train, batch_size=512)
    except KeyboardInterrupt:
        print("training interrupted.")

