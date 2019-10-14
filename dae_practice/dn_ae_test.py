from sklearn import decomposition
from matplotlib import pyplot as plt
import tensorflow as tf
import autoencoder_mnist as ae
import argparse, input_data
import numpy as np
# model-checkpoint-0349-191950

def corrupt_input(x):
    corrupting_matrix = tf.random_uniform(shape=tf.shape(x), minval=0,maxval=2,dtype=tf.int32)
    return x * tf.cast(corrupting_matrix, tf.float32)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test various optimization strategies')
    parser.add_argument('savepath', nargs=1, type=str)
    args = parser.parse_args()

    print("\nPULLING UP MNIST DATA")
    mnist = input_data.read_data_sets("data/", one_hot=False)
    print(mnist.test.labels)

    with tf.Graph().as_default():

        with tf.variable_scope("autoencoder_model"):

            x = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28=784
            corrupt = tf.placeholder(tf.float32)
            phase_train = tf.placeholder(tf.bool)

            c_x = (corrupt_input(x) * corrupt) + (x * (1 - corrupt))  

            code = ae.encoder(c_x, 2, phase_train)

            output = ae.decoder(code, 2, phase_train)

            cost, train_summary_op = ae.loss(output, x)

            global_step = tf.Variable(0, name='global_step', trainable=False)

            train_op = ae.training(cost, global_step)

            eval_op, in_im_op, out_im_op, val_summary_op = ae.evaluate(output, x)

            saver = tf.train.Saver()

            sess = tf.Session()


            print("\nSTARTING AUTOENCODER\n", args.savepath[0])
            sess = tf.Session()
            saver = tf.train.Saver()
            saver.restore(sess, args.savepath[0])

            print("\nGENERATING AE CODES AND RECONSTRUCTION")
            original_input, corr_func, noise_input, ae_reconstruction = sess.run([x, corrupt_input(x), c_x, output],feed_dict={x: mnist.test.images * np.random.randint(2, size=(784)), phase_train: True, corrupt: 1})

            plt.imshow(original_input[2].reshape((28,28)), cmap=plt.cm.gray)
            plt.show()
            plt.imshow(corr_func[2].reshape((28, 28)), cmap=plt.cm.gray)
            plt.show()
            plt.imshow(noise_input[2].reshape((28,28)), cmap=plt.cm.gray)
            plt.show()
            plt.imshow(ae_reconstruction[2].reshape((28,28)), cmap=plt.cm.gray)
            plt.show()
