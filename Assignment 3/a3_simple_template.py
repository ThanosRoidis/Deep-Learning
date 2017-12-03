import tensorflow as tf
import numpy as np
import time
from datetime import datetime

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


class NaiveBayesModel(object):

    def __init__(self, w_init, b_init = None, c_init = None):
        """
        :param w_init: An (n_categories, n_dim) array, where w[i, j] represents log p(X[j]=1 | Z[i]=1)
        :param b_init: A (n_categories, ) vector where b[i] represents log p(Z[i]=1), or None to fill with zeros
        :param c_init: A (n_dim, ) vector where b[j] represents log p(X[j]=1), or None to fill with zeros
        """

        self._n_categories = w_init.shape[0]
        self._n_dim = w_init.shape[1]

        if b_init is None:
            b_init =  np.zeros(self._n_categories, )
        if c_init is None:
            c_init = np.zeros(self._n_dim, )

        self.W = tf.get_variable('w', shape=w_init.shape, initializer=tf.constant_initializer(w_init))

        self.b = tf.get_variable('b', shape=b_init.shape, initializer=tf.constant_initializer(b_init))

        self.c = tf.get_variable('c', shape=c_init.shape, initializer=tf.constant_initializer(c_init))



    def log_p_x_given_z(self, x, z):
        """
        :param x: An (n_samples, n_dims) tensor
        :param z: An (n_samples, n_labels) tensor of integer class labels
        :return: An (n_samples, n_labels) tensor  p_x_given_z where result[i, j] indicates p(X=x[i] | Z=z[j])

        """

        #copy 'x' tensor 'n_categories' times along on the 3rd dimension [batch_size, n_dim]
        x_bc = tf.tile(tf.expand_dims(x, -1), [1, 1, self._n_categories])

        theta = tf.nn.sigmoid(tf.add(self.W, self.c))
        theta = tf.transpose(theta)

        log_probs = tf.log( x_bc * theta + (1-x_bc) * (1-theta))

        log_probs = tf.reduce_sum(log_probs, axis = 1)


        self.x_bc = x_bc
        self.theta = theta


        return log_probs



    def log_p_x(self, x):
        """
        :param x: A (n_samples, n_dim) array of data points
        :return: A (n_samples, ) array of log-probabilities assigned to each point
        """

        log_p_x_given_z = self.log_p_x_given_z(x, None)

        log_p_z = tf.nn.log_softmax(self.b)

        t = log_p_z + log_p_x_given_z

        return tf.reduce_logsumexp(t, axis = 1)



    def sample(self, n_samples):
        """
        :param n_samples: Generate N samples from your model
        :return: A (n_samples, n_dim) array where n_dim is the dimenionality of your input
        """

        cat_dist = tf.distributions.Categorical(logits=tf.nn.log_softmax(self.b))
        z_samples = cat_dist.sample(n_samples)

        bern_dist = tf.distributions.Bernoulli(probs = self.theta)
        samples = bern_dist.sample(n_samples)

        return samples, z_samples


def train_simple_generative_model_on_mnist(n_categories=20, initial_mag = 0.01, optimizer='rmsprop', learning_rate=.01, n_epochs=20, test_every=100,
                                           minibatch_size=100, plot_n_samples=16):
    """
    Train a simple Generative model on MNIST and plot the results.

    :param n_categories: Number of latent categories (K in assignment)
    :param initial_mag: Initial weight magnitude
    :param optimizer: The name of the optimizer to use
    :param learning_rate: Learning rate for the optimization
    :param n_epochs: Number of epochs to train for
    :param test_every: Test every X iterations
    :param minibatch_size: Number of samples in a minibatch
    :param plot_n_samples: Number of samples to plot
    """

    # minibatch_size = 2
    # n_categories = 3
    # Z = tf.placeholder(tf.float32, [None, n_categories])

    # Get Data
    x_train, x_test = load_mnist_images(binarize=True)
    train_iterator = tf.contrib.data.Dataset.from_tensor_slices(x_train).repeat().batch(minibatch_size).make_initializable_iterator()
    n_samples, n_dims = x_train.shape
    x_minibatch = train_iterator.get_next()  # Get symbolic data, target tensors

    print(n_samples, n_dims)
    print(type(x_minibatch))
    print(x_minibatch)

    # Build the model
    w_init = np.random.normal(scale = initial_mag, size = (n_categories, n_dims))
    b_init = np.random.normal(scale = initial_mag, size = (n_categories, ))
    c_init = np.zeros(n_dims, )

    model = NaiveBayesModel(w_init, b_init, c_init)

    loss = -tf.reduce_mean(model.log_p_x(x_minibatch))

    train_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

    # samples, z_samples = model.sample(plot_n_samples)

    log_path = './summaries/NB_' + time.strftime("%Y%m%d-%H%M")
    if not tf.gfile.Exists(log_path):
        tf.gfile.MakeDirs(log_path)
    writer = tf.summary.FileWriter(log_path)

    images_placeholder = tf.placeholder("float", [None,28,28,1], name='ims')
    im_summary = tf.summary.image('params',images_placeholder, max_outputs = 100)


    with tf.Session() as sess:
        sess.run(train_iterator.initializer)
        sess.run(tf.global_variables_initializer())
        n_steps =  int((n_epochs * n_samples)/minibatch_size)

        for i in range(n_steps):
            # Only for time measurement of step through network
            t1 = time.time()


            [_, loss_value] = sess.run([train_step, loss])


            # Only for time measurement of step through network
            t2 = time.time()
            examples_per_second = minibatch_size / float(t2 - t1)


            if i%test_every==0:

                print("[{}] Train Step {:04d}/{:04d}, "
                      "Examples/Sec = {:.2f}, Loss = {}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), i,
                    n_steps,  examples_per_second,
                    loss_value))


                theta_val = sess.run(model.theta)
                theta_val = theta_val.reshape((28,28,20))
                theta_val = np.transpose(theta_val, [2,0,1])
                theta_val = np.expand_dims(theta_val,axis = -1)

                # [samples_val, z_samples_val] = sess.run([samples,z_samples])

                # samples_val = samples_val[:, :, z_samples_val]

                # print(samples_val.shape)
                summary_result = sess.run(im_summary, feed_dict={images_placeholder:theta_val})

                writer.add_summary(summary_result, global_step=i)

                # raise NotImplementedError('INSERT CODE TO RUN TEST AND RECORD LOG-PROB PERIODICALLY')

            # raise NotImplementedError('COMPUTE TRAINING UPDATES HERE')


if __name__ == '__main__':
    train_simple_generative_model_on_mnist()

