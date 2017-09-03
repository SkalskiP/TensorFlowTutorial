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
