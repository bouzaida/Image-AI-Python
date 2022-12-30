

"""
Course:  Convolutional Neural Networks for Image Classification

Section-2
Process other datasets to use them for Classification

Description:
Modify images of MNIST and CIFAR to use them for classification
Assemble and save prepared datasets


File: process_mnist_dataset.py
"""


# Algorithm:
# --> Reading binary files
# --> Shuffling data along the first axis
# --> Splitting arrays into train and validation
# --> Saving arrays into HDF5 binary file
#
# Result: HDF5 binary file with MNIST dataset


# Importing needed libraries
import numpy as np
import h5py

from sklearn.utils import shuffle


"""
Start of:
Reading binary files of MNIST dataset
"""

# (!) On Windows, it might need to change
# this: '/'
# to this: '\'
# or to this: '\\'


# Reading binary file with images for training
# Opening file to read it in binary mode by 'rb'
with open('mnist/train-images.idx3-ubyte', 'rb') as bytestream:
    # Offset by the first 16 bytes
    bytestream.read(16)

    # Reading 60 000 images of 28x28 pixels each
    data = bytestream.read(60000 * 28 * 28)

    # Check point
    # Showing type of the data
    print(type(data))

    # Converting data into Numpy array
    x_train = np.frombuffer(data, dtype=np.uint8).astype(np.float32)

    # Reshaping data to make every image as separate matrix of 28x28 pixels
    x_train = x_train.reshape(60000, 28, 28, 1)

    # Check points
    # Showing type of the data after conversion
    print(type(x_train))

    # Showing shape of the data
    print(x_train.shape)
    print()


# Reading binary file with class indexes of images for training
# Opening file to read it in binary mode by 'rb'
with open('mnist/train-labels.idx1-ubyte', 'rb') as bytestream:
    # Offset by the first 8 bytes
    bytestream.read(8)

    # Reading 60 000 class indexes
    data = bytestream.read(60000)

    # Check point
    # Showing type of the data
    print(type(data))

    # Converting data into Numpy array
    y_train = np.frombuffer(data, dtype=np.uint8)

    # Check points
    # Showing type of the data after conversion
    print(type(y_train))

    # Showing shape of the data
    print(y_train.shape)
    print()


# Reading binary file with images for testing
# Opening file to read it in binary mode by 'rb'
with open('mnist/t10k-images.idx3-ubyte', 'rb') as bytestream:
    # Opening file to read it in binary mode by 'rb'
    bytestream.read(16)

    # Reading 10 000 images of 28x28 pixels each
    data = bytestream.read(10000 * 28 * 28)

    # Check point
    # Showing type of the data
    print(type(data))

    # Converting data into Numpy array
    x_test = np.frombuffer(data, dtype=np.uint8).astype(np.float32)

    # Reshaping data to make every image as separate matrix of 28x28 pixels
    x_test = x_test.reshape(10000, 28, 28, 1)

    # Check points
    # Showing type of the data after conversion
    print(type(x_test))

    # Showing shape of the data
    print(x_test.shape)
    print()


# Reading binary file with class indexes of images for testing
# Opening file to read it in binary mode by 'rb'
with open('mnist/t10k-labels.idx1-ubyte', 'rb') as bytestream:
    # Offset by the first 8 bytes
    bytestream.read(8)

    # Reading 10 000 class indexes
    data = bytestream.read(10000)

    # Check point
    # Showing type of the data
    print(type(data))

    # Converting data into Numpy array
    y_test = np.frombuffer(data, dtype=np.uint8)

    # Check points
    # Showing type of the data after conversion
    print(type(y_test))

    # Showing shape of the data
    print(y_test.shape)

"""
End of:
Reading binary files of MNIST dataset
"""


"""
Start of:
Shuffling data along the first axis
"""

# Shuffling data along the first axis
# Saving appropriate connection: image --> label
x_train, y_train = shuffle(x_train, y_train)

"""
End of:
Shuffling data along the first axis
"""


"""
Start of:
Splitting arrays into train and validation
"""

# Slicing first 15% of elements from Numpy arrays for training
# Assigning sliced elements to validation Numpy arrays
x_validation = x_train[:int(x_train.shape[0] * 0.15), :, :, :]
y_validation = y_train[:int(y_train.shape[0] * 0.15)]


# Slicing last 85% of elements from Numpy arrays for training
# Re-assigning sliced elements to train Numpy arrays
x_train = x_train[int(x_train.shape[0] * 0.15):, :, :, :]
y_train = y_train[int(y_train.shape[0] * 0.15):]

"""
End of:
Splitting arrays into train, validation and test
"""


"""
Start of:
Saving arrays into HDF5 binary file
"""

# Saving prepared Numpy arrays into HDF5 binary file
# Initiating File object
# Creating file with name 'dataset_mnist.hdf5'
# Opening it in writing mode by 'w'
with h5py.File('dataset_mnist.hdf5', 'w') as f:
    # Calling methods to create datasets of given shapes and types
    # Saving Numpy arrays for training
    f.create_dataset('x_train', data=x_train, dtype='f')
    f.create_dataset('y_train', data=y_train, dtype='i')

    # Saving Numpy arrays for validation
    f.create_dataset('x_validation', data=x_validation, dtype='f')
    f.create_dataset('y_validation', data=y_validation, dtype='i')

    # Saving Numpy arrays for testing
    f.create_dataset('x_test', data=x_test, dtype='f')
    f.create_dataset('y_test', data=y_test, dtype='i')

"""
End of:
Saving arrays into HDF5 binary file
"""


"""
Some comments
Function 'np.frombuffer' interprets a buffer as a 1-dimensional array.

More details and examples are here:
print(help(np.frombuffer))
https://numpy.org/doc/stable/reference/generated/numpy.frombuffer.html

"""
