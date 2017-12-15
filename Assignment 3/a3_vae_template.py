import numpy as np
import tensorflow as tf

import time
from datetime import datetime
import random
import matplotlib.pyplot as plt
from scipy.misc import imresize



def load_mnist_images(binarize=True):
    """
    :param binarize: Turn the images into binary vectors
    :return: x_train, x_test  Where
        x_train is a (55000 x 784) tensor of training images
        x_test is a  (10000 x 784) tensor of test images
    """
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
    x_train = mnist.train.images
    x_test = mnist.test.images
    if binarize:
        x_train = (x_train>0.5).astype(x_train.dtype)
        x_test = (x_test>0.5).astype(x_test.dtype)
    return x_train, x_test


class VariationalAutoencoder(object):


    def __init__(self, x, z_dim, encoder_hidden_sizes, decoder_hidden_sizes, kernel_initializer):

        n_input =  x.get_shape()[1].value

        self.n_input = n_input
        self.z_dim = z_dim
        self.encoder_hidden_sizes = encoder_hidden_sizes
        self.decoder_hidden_sizes = decoder_hidden_sizes
        self.kernel_initializer = kernel_initializer



        #Define encoder
        self.encoder_mu, self.encoder_sigma = self.build_encoder(x, encoder_hidden_sizes, z_dim)

        #Sample with the reparametarization trick (1 sample only)
        z = self.encoder_mu + self.encoder_sigma * tf.random_normal(tf.shape(self.encoder_sigma), 0, 1, dtype=tf.float32)

        #Define decoder
        self.y = self.build_decoder(z, decoder_hidden_sizes, n_input)

        mu = self.encoder_mu
        sigma = self.encoder_sigma
        y = self.y
        self.z = z

        marginal_likelihood = tf.reduce_sum(x * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) +
                                            (1 - x) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)), 1)

        KL_divergence = 0.5 * tf.reduce_sum(tf.square(sigma) + tf.square(mu) -  tf.log(tf.clip_by_value(tf.square(sigma), 1e-10, 1.0)) - 1, 1)


        ELBO = tf.reduce_mean(marginal_likelihood - KL_divergence)


        self.ELBO = ELBO
        self.loss = -ELBO

    def build_encoder(self, x, n_hidden, n_output):

        with tf.variable_scope('encoder') as scope:
            encoder_layers = [x]

            for l in range(len(n_hidden)):

                layer = tf.layers.dense(inputs=encoder_layers[-1], units = n_hidden[l],
                                        activation=tf.nn.relu,
                                        kernel_initializer=self.kernel_initializer, name = 'dense_%d'%l)

                encoder_layers.append(layer)


            mu = tf.layers.dense(inputs=encoder_layers[-1], units = n_output,
                                        activation= None,
                                        kernel_initializer=self.kernel_initializer, name='mu')


            sigma = tf.layers.dense(inputs=encoder_layers[-1], units = n_output,
                                        activation=tf.nn.softplus,
                                        kernel_initializer=self.kernel_initializer,name='sigma')



        return mu, sigma


    def build_decoder(self, z, n_hidden, n_output):

        with tf.variable_scope('decoder') as scope:

            decoder_layers = [z]

            for l in range(len(n_hidden)):

                layer = tf.layers.dense(decoder_layers[-1], n_hidden[l],
                                        activation=tf.nn.relu,
                                        kernel_initializer=self.kernel_initializer, name = 'dense_%d'%l)

                decoder_layers.append(layer)

            output_bern = tf.layers.dense(decoder_layers[-1], n_output,
                                        activation=tf.nn.sigmoid,
                                        kernel_initializer=self.kernel_initializer, name = 'dense_out')

        return output_bern



    def lower_bound(self, x):
        """
        :param x: A (n_samples, n_dim) array of data points
        :return: A (n_samples, ) array of the lower-bound on the log-probability of each data point
        """

        pass

    def mean_x_given_z(self, z):
        """
        :param z: A (n_samples, n_dim_z) tensor containing a set of latent data points (n_samples, n_dim_z)
        :return: A (n_samples, n_dim_x) tensor containing the mean of p(X|Z=z) for each of the given points
        """

        # Define decoder
        y = self.build_decoder(z, self.decoder_hidden_sizes, self.n_input)

        return y


    def sample(self, n_samples):
        """
        :param n_samples: Generate N samples from your model
        :return: A (n_samples, n_dim) array where n_dim is the dimenionality of your input
        """

        tf.random_normal( shape = [n_samples, self.z_dim])


    def sample_Z(self, n_samples):
        """
        :param n_samples: Generate N samples from your model
        :return: A (n_samples, n_dim) array where n_dim is the dimenionality of your input
        """

        samples_z = tf.random_normal( shape = [n_samples, self.z_dim])

        return samples_z


    def sample_X(self):
        """
        :param n_samples: Generate N samples from your model
        :return: A (n_samples, n_dim) array where n_dim is the dimenionality of your input
        """

        bern = tf.distributions.Bernoulli(probs = self.y)
        sample_X = bern.sample(1)
        sample_X = tf.squeeze((sample_X))
        return sample_X


def subplot_digits(digits, labels, shape):

    f, axarr = plt.subplots(*shape, figsize = (25,25))
    axarr = axarr.flat

    for i in range(digits.shape[0]):

        digit = digits[i,:].reshape((28, 28))

        # (x,y) = (i//shape[1], i % shape[1])

        axarr[i].imshow(digit, cmap='gray')
        # axarr[i].set_title(labels[i])
        axarr[i].axis('off')


    # f.subplots_adjust(hspace=0.3)
    f.tight_layout(pad=0, w_pad=0, h_pad=0)

    plt.draw()


def merge_images(flat_images, size, resize_factor = 1.0):
    # flat_images is an Nx784 array, combines all of the images into an 28*sqrt(N) x 28*sqrt(N) array

    images = flat_images.reshape(-1, 28, 28)

    h, w = 28,28

    h_ = int(h * resize_factor)
    w_ = int(w * resize_factor)

    img = np.zeros((h_ * size[0], w_ * size[1]))

    for idx, image in enumerate(images):
        i = int(idx % size[1])
        j = int(idx / size[1])

        image_ = imresize(image.squeeze(), size=(w_, h_), interp='bicubic')

        img[j * h_:j * h_ + h_, i * w_:i * w_ + w_] = image_

    return img

def plot_manifold(sess, model, z_lim, manifold_size):

    x = np.linspace(-z_lim, z_lim, manifold_size)
    y = np.linspace(-z_lim, z_lim, manifold_size)
    xv, yv = np.meshgrid(x, y)

    xv = xv.reshape(manifold_size ** 2, 1)
    yv = yv.reshape(manifold_size ** 2, 1)

    manifold_Zs = np.concatenate((xv, yv), axis=1)

    [manifold_samples] = sess.run([model.y], feed_dict={model.z: manifold_Zs})


    img = merge_images(manifold_samples, size = (manifold_size, manifold_size))


    plt.imshow(img, cmap = 'gray')
    plt.axis('off')


def plot_manifold2(sess, model, z_lim, manifold_size):

    x = np.linspace(-z_lim, z_lim, manifold_size)
    y = np.linspace(-z_lim, z_lim, manifold_size)
    xv, yv = np.meshgrid(x, y)

    xv = xv.reshape(manifold_size ** 2, 1)
    yv = yv.reshape(manifold_size ** 2, 1)

    manifold_Zs = np.concatenate((xv, yv), axis=1)

    [manifold_samples] = sess.run([model.y], feed_dict={model.z: manifold_Zs})



    resize_factor = 1.0
    size = (manifold_size,manifold_size)

    images = manifold_samples.reshape(-1,28,28)

    h, w = images.shape[1], images.shape[2]

    h_ = int(h * resize_factor)
    w_ = int(w * resize_factor)

    img = np.zeros((h_ * size[0], w_ * size[1]))

    for idx, image in enumerate(images):
        i = int(idx % size[1])
        j = int(idx / size[1])

        image_ = imresize(image.squeeze(), size=(w_, h_), interp='bicubic')

        img[j * h_:j * h_ + h_, i * w_:i * w_ + w_] = image_


    plt.imshow(img, cmap = 'gray')


def plot_scattered_image(self, z, id, name='scattered_image.jpg'):
    N = 10
    # z_range = 4
    plt.figure(figsize=(8, 6))
    plt.scatter(z[:, 0], z[:, 1], c=np.argmax(id, 1), marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
    plt.colorbar(ticks=range(N))
    axes = plt.gca()
    axes.set_xlim([-self.z_range - 2, self.z_range + 2])
    axes.set_ylim([-self.z_range - 2, self.z_range + 2])
    plt.grid(True)
    #plt.savefig(self.DIR + "/" + name)


def train_vae_on_mnist(z_dim=2, kernel_initializer='glorot_uniform', optimizer = 'adam',  learning_rate=0.001, n_epochs=40,
        test_every=100, minibatch_size=100, encoder_hidden_sizes=[200, 200], decoder_hidden_sizes=[200, 200],
        hidden_activation='relu', plot_grid_size=10, plot_n_samples = 20):
    """
    Train a variational autoencoder on MNIST and plot the results.

    :param z_dim: The dimensionality of the latent space.
    :param kernel_initializer: How to initialize the weight matrices (see tf.keras.layers.Dense)
    :param optimizer: The optimizer to use
    :param learning_rate: The learning rate for the optimizer
    :param n_epochs: Number of epochs to train
    :param test_every: Test every X training iterations
    :param minibatch_size: Number of samples per minibatch
    :param encoder_hidden_sizes: Sizes of hidden layers in encoder
    :param decoder_hidden_sizes: Sizes of hidden layers in decoder
    :param hidden_activation: Activation to use for hidden layers of encoder/decoder.
    :param plot_grid_size: Number of rows, columns to use to make grid-plot of images corresponding to latent Z-points
    :param plot_n_samples: Number of samples to draw when plotting samples from model.
    """


    # Get Data
    x_train, x_test = load_mnist_images(binarize=True)

    train_iterator = tf.contrib.data.Dataset.from_tensor_slices(x_train).repeat().batch(minibatch_size).make_initializable_iterator()
    n_samples, n_dims = x_train.shape
    x_minibatch = train_iterator.get_next()  # Get symbolic data, target tensors

    # Build Model
    if kernel_initializer == 'glorot_uniform':
        kernel_initializer = tf.contrib.layers.xavier_initializer()
    else:
        kernel_initializer = tf.contrib.layers.xavier_initializer()

    model = VariationalAutoencoder(x_minibatch, z_dim, encoder_hidden_sizes, decoder_hidden_sizes, kernel_initializer)
    loss = model.loss

    if optimizer == "adam":
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    else:
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # Define the optimizer
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # grads_and_vars = optimizer.compute_gradients(model.loss)
    # grads, variables = zip(*grads_and_vars)
    # grads_clipped, _ = tf.clip_by_global_norm(grads, clip_norm=10)
    # train_step = optimizer.apply_gradients(zip(grads_clipped, variables))

    # Setup summaries and writers
    log_path = './summaries/VAE_' + time.strftime("%Y%m%d-%H%M")
    if not tf.gfile.Exists(log_path):
        tf.gfile.MakeDirs(log_path)
        tf.gfile.MakeDirs(log_path + '/train')
    writer = tf.summary.FileWriter(log_path)
    train_writer = tf.summary.FileWriter(log_path + '/train')

    loss_sum = tf.summary.scalar('Loss', loss)

    with tf.Session() as sess:
        sess.run(train_iterator.initializer)
        sess.run(tf.global_variables_initializer())
        n_steps =  (n_epochs * n_samples)//minibatch_size

        writer.add_graph(sess.graph)

        for i in range(n_steps):
            # Only for time measurement of step through network
            t1 = time.time()

            # [_, sum] = sess.run([train_step, merged_summary])
            # writer.add_summary(sum, global_step=i)


            [_] = sess.run([train_step])


            # Only for time measurement of step through network
            t2 = time.time()
            examples_per_second = minibatch_size / float(t2 - t1)

            if i%test_every==0:

                test_size = x_test.shape[0]

                test_loss = sess.run(loss, feed_dict={x_minibatch: x_test[:test_size]})

                train_loss = sess.run(loss, feed_dict={x_minibatch: x_train[:test_size]})

                print("[{}] Train Step {:04d}/{:04d}, "
                      "Examples/Sec = {:.2f}, Loss = {} | {}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), i,
                    n_steps,  examples_per_second,
                    train_loss, test_loss))

                # write train and test loss
                train_writer.add_summary(sess.run(loss_sum, feed_dict={loss: train_loss}), global_step=i)

                writer.add_summary(sess.run(loss_sum, feed_dict={loss: test_loss}), global_step=i)


        sample_Z_op = model.sample_Z(10)
        sample_X_op = model.sample_X()

        [z_samples] = sess.run([sample_Z_op])
        [x_samples] = sess.run([sample_X_op], feed_dict={model.z: z_samples})


        [mean_x_samples] = sess.run([model.y], feed_dict={model.z: z_samples})

        labels = ["z = {}".format(z) for z in z_samples]
        # subplot_digits(x_samples, labels, shape = (5,2))
        img = merge_images(x_samples, size = (2,5))
        plt.figure(1)
        plt.imshow(img, cmap = 'gray')
        plt.axis("off")


        # subplot_digits(mean_x_samples, labels, shape = (5,2))
        img = merge_images(mean_x_samples, size=(2, 5))
        plt.figure(2)
        plt.imshow(img, cmap='gray')
        plt.axis("off")


        plt.figure(3)
        plot_manifold(sess, model, z_lim = 2, manifold_size = 20)

        plt.show()




if __name__ == '__main__':
    train_vae_on_mnist(n_epochs = 1)