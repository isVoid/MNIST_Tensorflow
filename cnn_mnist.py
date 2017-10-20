from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    # MNIST input DIM = 28x28

    # Conv Layer 1
    conv1 = tf.layers.conv2d(
        inputs=input_layer, # Input of layer one is input_layer (x)
        filters=32,         # 32 filters of first layer
        kernel_size=[5, 5], # Kernal size is 5x5
        padding="same",     # The next layer will have the same width and height with the padding
        activation=tf.nn.relu # Activation function is relu
    )
    # H1 output DIM = 28x28x32

    # Pooling Layer 1, max pooling
    pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size =[2, 2], strides = 2)
    # Pooling layer DIM = 14x14x32

    # Conv Layer 2
    conv2 = tf.layers.conv2d(
        inputs = pool1,     # Input layer is pooling layer 1
        filters=64,         # 64 filters of second conv layer
        kernel_size = [5, 5], # kernal size is 5x5
        padding="same",     # "SAME" dim output
        activation=tf.nn.relu #Activation func is relu
    )
    # Conv2 output DIM = 14x14x64, notice the depth

    # Pooling Layer 2, max_pooling
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides = 2)
    # Hidden layer 2 output DIM = 7x7x64

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units = 1024, activation=tf.nn.relu)
    # Introduce dropout, rate 0.4
    # Which means for dense layer, during training, 40% of element will be randomly dropped out
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN
    )

    # logits layer
    # Contains RAW values of the predictions, with 10 neurons, representing 10 classes (0~9)
    logits = tf.layers.dense(inputs=dropout, units=10)

    # We want the network to preduce:
    # 1. Prediction for each sample
    # 2. probabilities for each prediction
    predictions = {
        # Generate predictions
        "classes" : tf.argmax(input=logits, axis=1),
        # Add 'softmax_tensor' to the graph. used for predict and by the logging_hook
        "probabilities" : tf.nn.softmax(logits, name="softmax_tensor")
    }

    # Compiling Estimator Specification in PREDICT mode, the compiled
    # object is type EstimatorSpec
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions=predictions)

    # Calculate Loss (for both train and eval)
    # One-hot encoding is like follows:
    # Given a classification of 5 classes, if the correct labels is 2,
    # it should be encoded as [0, 0, 1, 0, 0]
    onehot_labels = tf.one_hot(indices = tf.cast(labels, tf.int32), depth=10)   #onehot_labels contains the one_hot encoding of ground-truth labels
    loss = tf.losses.softmax_cross_entropy(                                     #Cross-Entropy loss for this prediction
        onehot_labels=onehot_labels, logits=logits
    )

    # Configure Training Op:
    # Use SGD of learning rate 0.001
    # To Minimize Target "loss"
    # return an estimatorSpec
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss = loss,
            global_step = tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evalulation metrics (for EVAL mode)
    # Metric is accuracy, comparing ground truth and predictions
    eval_metric_ops = {
        "accuracy" : tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]
        )
    }
    return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    # Load Training Data:
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    # Conform traning data to np.arrays
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    # Conform eval (test) data to np.arrays
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Create Estimator
    # Estimator is a high level Tensorflow class for model training, evalulation
    # and inference
    mnist_classifier = tf.estimator.Estimator(
        # What model to use
        model_fn = cnn_model_fn,
        # What directory to save temp data
        model_dir="/Users/WangYinghao/Documents/pythonPlayground/TensorflowPlayground/mnist/model/"
    )

    # Setting up logging hook to see training status
    tensors_to_log = {"probabilities" : "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        # Log every 50 iterations
        tensors=tensors_to_log, every_n_iter=50
    )

    # Train the model!
    # Set train_x, train_y, batch GD size, enable shuffle
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None, # this means the model will train till specified step.
        shuffle=True
    )

    # Don't forget to setup logging hooks!
    mnist_classifier.train(input_fn = train_input_fn, steps=20000,
        hooks=[logging_hook])

    # Evaluate model
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x" : eval_data},
        y = eval_labels,
        num_epochs = 1, # this means the model will only run over evaluate data for 1 time
        shuffle=False
    )
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print (eval_results)

if __name__ == "__main__":
    tf.app.run()
