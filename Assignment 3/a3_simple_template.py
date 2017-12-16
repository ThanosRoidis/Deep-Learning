import tensorflow as tf
import numpy as np
import time
from datetime import datetime
import random
import matplotlib.pyplot as plt

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



    def sample(self, n_samples, sess = None):
        """
        :param n_samples: Generate N samples from your model
        :return: A (n_samples, n_dim) array where n_dim is the dimenionality of your input
        """

        cat_dist = tf.distributions.Categorical(logits=tf.nn.log_softmax(self.b))
        Z_samples = cat_dist.sample(n_samples)


        sample_thetas = tf.gather(self.theta,Z_samples,axis = 1)

        X_samples = tf.distributions.Bernoulli(probs = sample_thetas).sample(1)
        X_samples = tf.transpose(tf.squeeze(X_samples))


        return X_samples, Z_samples



def subplot_digits(digits, labels, shape, title = ""):

    f, axarr = plt.subplots(*shape, figsize=(15,15))
    f.canvas.set_window_title(title)
    axarr = axarr.flat

    for i in range(digits.shape[0]):

        digit = digits[i,:].reshape((28, 28))

        # (x,y) = (i//shape[1], i % shape[1])

        axarr[i].imshow(digit, cmap='gray')
        axarr[i].set_title(labels[i])
        axarr[i].axis('off')

    # f.subplots_adjust(hspace=0.3)
    # f.tight_layout(pad=0, w_pad=0, h_pad=0)
    f.tight_layout()

    plt.draw()
    #f.savefig("./{}.png".format(title.replace(" ", "_")))


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


def get_frankenstein_digits(x_train, frankenstein_size):

    original_images = x_train[:frankenstein_size]

    frankenstein_images = np.copy(original_images)
    indexes = np.random.choice(np.arange(frankenstein_size, 2 * frankenstein_size), frankenstein_size, replace=False)

    second_half_images = x_train[indexes,:]

    for i in range(frankenstein_size):
        fr_im = frankenstein_images[i].reshape(28,28)
        sd_im =  second_half_images[i].reshape(28,28)

        fr_im[:, 28 // 2:] = sd_im[:, 28 // 2:]
        frankenstein_images[i] = fr_im.flat

    return frankenstein_images, original_images


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

    random.seed(42)

    # Get Data
    x_train, x_test = load_mnist_images(binarize=True)
    train_iterator = tf.contrib.data.Dataset.from_tensor_slices(x_train).repeat().batch(minibatch_size).make_initializable_iterator()
    n_samples, n_dims = x_train.shape
    x_minibatch = train_iterator.get_next()  # Get symbolic data, target tensors


    # Build the model
    w_init = np.random.normal(scale = initial_mag, size = (n_categories, n_dims))
    b_init = np.random.normal(scale = initial_mag, size = (n_categories, ))
    c_init = np.zeros(n_dims, )

    model = NaiveBayesModel(w_init, b_init, c_init)

    log_p_x = model.log_p_x(x_minibatch)

    loss = -tf.reduce_mean(log_p_x)

    train_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

    # Setup summaries and writers
    log_path = './summaries/NB_' + time.strftime("%Y%m%d-%H%M")
    if not tf.gfile.Exists(log_path):
        tf.gfile.MakeDirs(log_path)
        tf.gfile.MakeDirs(log_path + '/train')
    writer = tf.summary.FileWriter(log_path)
    train_writer = tf.summary.FileWriter(log_path + '/train')

    images_placeholder = tf.placeholder("float", [None,28,28,1], name='images')
    exp_pix_summary = tf.summary.image('params',images_placeholder, max_outputs = 100)

    log_prob_summ = tf.summary.scalar('Log_prob', -loss)


    with tf.Session() as sess:
        sess.run(train_iterator.initializer)
        sess.run(tf.global_variables_initializer())
        n_steps =  (n_epochs * n_samples)//minibatch_size

        writer.add_graph(sess.graph)




        for i in range(n_steps):
            # Only for time measurement of step through network
            t1 = time.time()


            sess.run(train_step)


            # Only for time measurement of step through network
            t2 = time.time()
            examples_per_second = minibatch_size / float(t2 - t1)


            if i%test_every==0:

                test_size = x_test.shape[0]

                test_log_prob = -sess.run(loss, feed_dict={x_minibatch: x_test[:test_size]})

                train_log_prob = -sess.run(loss, feed_dict={x_minibatch: x_train[:test_size]})

                print("[{}] Train Step {:04d}/{:04d}, "
                      "Examples/Sec = {:.2f}, Log Prob = {:.8f} | {:.8f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), i,
                    n_steps,  examples_per_second,
                    train_log_prob, test_log_prob))


                # Summary the expected pixel values per category
                theta_val = sess.run(model.theta)
                theta_val = theta_val.reshape((28,28,20))
                theta_val = np.transpose(theta_val, [2,0,1])
                theta_val = np.expand_dims(theta_val,axis = -1)

                writer.add_summary(sess.run(exp_pix_summary, feed_dict={images_placeholder:theta_val}), global_step=i)

                # write train and test loss
                train_writer.add_summary(sess.run(log_prob_summ, feed_dict={loss: -train_log_prob}), global_step=i)

                writer.add_summary(sess.run(log_prob_summ, feed_dict={loss: -test_log_prob}), global_step=i)




        # Plot the expected pixel values per category
        theta_val = sess.run(model.theta)
        theta_val = np.transpose(theta_val)
        labels = ["Category = {}".format(z) for z in range(n_categories)]
        subplot_digits(theta_val, labels, shape = (5,4), title ="Expected pixel values per category")


        # Sample and plot 'plot_n_samples' images
        [X_samples, Z_samples] = sess.run(model.sample(plot_n_samples))
        labels = ["Category = {}".format(z) for z in Z_samples]
        subplot_digits(X_samples, labels, shape = (plot_n_samples // 4,4),title ="Sampled Images")


        #Create and plot 10 frankenstein images, along with the original
        frankenstein_size = 10

        frankenstein_images,original_images = get_frankenstein_digits(x_train, frankenstein_size)

        or_log_p_x = sess.run(log_p_x, feed_dict={x_minibatch: original_images})
        fr_log_p_x = sess.run(log_p_x, feed_dict={x_minibatch: frankenstein_images})

        labels = ["logp = {0:.2f}".format(p) for p in fr_log_p_x]
        subplot_digits(frankenstein_images, labels, shape = (2,5),title ="Frankenstein Images")
        print("Frankenstein mean log prob: {}".format(np.mean(fr_log_p_x)))

        labels = ["logp = {0:.2f}".format(p) for p in or_log_p_x]
        subplot_digits(original_images, labels, shape = (2,5),title ="Original Frankenstein Images")
        print("Original Frankenstein mean log prob: {}".format(np.mean(or_log_p_x)))

        plt.show()




if __name__ == '__main__':

    if not tf.gfile.Exists('./summaries'):
        tf.gfile.MakeDirs('./summaries')
    train_simple_generative_model_on_mnist(plot_n_samples=16)

