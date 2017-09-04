# TensorFlow Tutorial #01
# Simple Linear Model

# GitHub URL: https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/01_Simple_Linear_Model.ipynb
# YouTube URL: https://www.youtube.com/watch?v=wuo4JdG3SvU&list=PL9Hr9sNUjfsmEu1ZniY0XpHSzl5uihcXZ
# TensorFlow URL: https://www.tensorflow.org/get_started/mnist/beginners

# Imports

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

# Load MNIST Data
###############################################################################

# MNIST is a simple computer vision dataset. It consists of images of handwritten digits.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# We can now check if mnist dataset was loaded correctly by checking sizes of subsets
"""
print("Size of:")
print("- Training-set:\t\t{}".format(len(mnist.train.labels)))
print("- Test-set:\t\t{}".format(len(mnist.test.labels)))
print("- Validation-set:\t{}".format(len(mnist.validation.labels)))
"""

# One-Hot Encoding
###############################################################################
"""
The data-set has been loaded as so-called One-Hot encoding. 
This means the labels have been converted from a single number to a vector 
whose length equals the number of possible classes. 
All elements of the vector are zero except for the $i$'th element which is one and means the class is $i$. 
"""

# One-Hot encoded labels for the first 5 images in the test-set are:
mnist.test.labels[0:5, :]

# Classes as single numbers
mnist.test.cls = np.array([label.argmax() for label in mnist.test.labels])

# Classes for the first five images in the test-set.
mnist.test.cls[0:5]

# Data dimensions
###############################################################################

# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of classes, one class for each of 10 digits.
num_classes = 10

# Helper-function for plotting images
###############################################################################

def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
        
# Plot a few images to see if data is correct
###############################################################################

# Get the first images from the test-set.
images = mnist.test.images[0:9]

# Get the true classes for those images.
cls_true = mnist.test.cls[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true)        

# Placeholder variables
###############################################################################
"""
First we define the placeholder variable for the input images.
This is a so-called tensor, which just means that it is a multi-dimensional vector or matrix.
The data-type is set to float32 and the shape is set to [None, img_size_flat],
 where None means that the tensor may hold an arbitrary number of images with each image being a vector of length img_size_flat.
"""
x = tf.placeholder(tf.float32, [None, img_size_flat])
"""
Next we have the placeholder variable for the true labels associated with the images that were input in the placeholder variable x. 
The shape of this placeholder variable is [None, num_classes] which means it may hold an arbitrary number of labels 
and each label is a vector of length num_classes which is 10 in this case.
"""
y_true = tf.placeholder(tf.float32, [None, num_classes])
"""
Finally we have the placeholder variable for the true class of each image in the placeholder variable x.
"""
y_true_cls = tf.placeholder(tf.int64, [None])

# Variables to be optimized
###############################################################################
"""
We also need the weights and biases for our model.
It is TensorFlow variable that must be initialized with zeros 
and whose shape is [img_size_flat, num_classes], 
so it is a 2-dimensional tensor (or matrix) with img_size_flat rows and num_classes columns.
"""
weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))
"""
The second variable that must be optimized is called biases 
and is defined as a 1-dimensional tensor (or vector) of length num_classes.
"""

biases = tf.Variable(tf.zeros([num_classes]))

# Model
###############################################################################
"""
This simple mathematical model multiplies the images in the placeholder variable x with the weights and then adds the biases.
"""
logits = tf.matmul(x, weights) + biases
y_pred = tf.nn.softmax(logits)
"""
The predicted class can be calculated from the y_pred matrix by taking the index of the largest element in each row.
"""
y_pred_cls = tf.argmax(y_pred, axis=1)

# Cost-function to be optimized
###############################################################################
"""
TensorFlow has a built-in function for calculating the cross-entropy. 
Note that it uses the values of the logits because it also calculates the softmax internally.
"""
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)

# Optimization method & Performance measures
###############################################################################
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)
"""
This is a vector of booleans whether the predicted class equals the true class of each image.
"""
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
"""
This calculates the classification accuracy by first type-casting the vector of booleans to floats, 
so that False becomes 0 and True becomes 1, and then calculating the average of these numbers.
"""
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# TensorFlow Run
###############################################################################
# Create TensorFlow session
session = tf.Session()
# Initialize variables
session.run(tf.global_variables_initializer())
# Batch size
batch_size = 100

# Helper-function to perform optimization iterations
def optimize(num_iterations):
    for i in range(num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = mnist.train.next_batch(batch_size)
        
        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        # Note that the placeholder for y_true_cls is not set
        # because it is not used during training.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)
        

# Helper-functions to show performance
"""
Dict with the test-set data to be used as input to the TensorFlow graph. 
Note that we must use the correct names for the placeholder variables in the TensorFlow graph.
"""
feed_dict_test = {x: mnist.test.images,
                  y_true: mnist.test.labels,
                  y_true_cls: mnist.test.cls}

# Function for printing the classification accuracy on the test-set.
def print_accuracy():
    # Use TensorFlow to compute the accuracy.
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    
    # Print the accuracy.
    print("Accuracy on test-set: {0:.1%}".format(acc))
    
# unction for printing and plotting the confusion matrix using scikit-learn.
def print_confusion_matrix():
    # Get the true classifications for the test-set.
    cls_true = mnist.test.cls
    
    # Get the predicted classifications for the test-set.
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # Make various adjustments to the plot.
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
# Function for plotting examples of images from the test-set that have been mis-classified.  
def plot_example_errors():
    # Use TensorFlow to get a list of boolean values
    # whether each test-image has been correctly classified,
    # and a list for the predicted class of each image.
    correct, cls_pred = session.run([correct_prediction, y_pred_cls],
                                    feed_dict=feed_dict_test)

    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = mnist.test.images[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = mnist.test.cls[incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])
    
# Performance before any optimization
print_accuracy()
plot_example_errors()

# Performance after 1 optimization iteration
