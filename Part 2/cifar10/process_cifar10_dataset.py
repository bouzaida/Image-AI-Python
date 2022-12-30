

"""
Course:  Convolutional Neural Networks for Image Classification

Section-2
Process other datasets to use them for Classification

Description:
Modify images of MNIST and CIFAR to use them for classification
Assemble and save prepared datasets


File: process_cifar10_dataset.py
"""


# Algorithm:
# --> Defining Numpy arrays to collect images
# --> Reading binary files
# --> Shuffling data along the first axis
# --> Splitting arrays into train and validation
# --> Saving arrays into HDF5 binary file
#
# Result: HDF5 binary file with CIFAR-10 dataset


# Importing needed libraries
import numpy as np
import h5py

from sklearn.utils import shuffle


"""
Start of:
Defining Numpy arrays to collect images
"""

# Preparing zero-valued Numpy array for cut objects
# Shape: image number, height, width, number of channels
x_train = np.zeros((1, 32, 32, 3))


# Preparing zero-valued Numpy array for class indexes
# Shape: class's number
y_train = np.zeros(1)


"""
Start of:
Defining Numpy arrays to collect images
"""


"""
Start of:
Reading binary files of CIFAR-10 dataset
"""

# (!) On Windows, it might need to change
# this: '/'
# to this: '\'
# or to this: '\\'


# Reading 5 binary files with images and class indexes for training
for i in range(1, 6):
    # Opening current file to read it in binary mode by 'rb'
    with open('cifar10/data_batch_' + str(i) + '.bin', 'rb') as bytestream:
        # Reading 10 000 images of 32x32 pixels each
        # Every image has its class index at the beginning
        data = bytestream.read(30730000)

        # Check point
        # Showing type of the data
        print(type(data))

        # Converting data into Numpy array
        data = np.frombuffer(data, dtype=np.uint8)

        # Check point
        # Showing type of the data after conversion
        print(type(data))

        # Reshaping data to 10 000 separate rows
        data = data.reshape(10000, 3073)

        # Extracting class indexes from every row
        y_temp = data[:, 0:1]

        # Reshaping Numpy array with class indexes
        y_temp = y_temp.reshape(10000)

        # Extracting images of 32x32 pixels each from every row
        x_temp = data[:, 1:].astype(np.float32)

        # Reshaping Numpy array with images
        # to make every image as separate matrix of 32x32 pixels and 3 channels
        x_temp = x_temp.reshape(10000, 3, 32, 32)

        # Making channels come at the end
        x_temp = x_temp.transpose(0, 2, 3, 1)

        # Check points
        # Showing shapes of the data
        print(x_temp.shape)
        print(y_temp.shape)
        print()

    # Concatenating vertically temp arrays to main arrays
    x_train = np.concatenate((x_train, x_temp), axis=0)
    y_train = np.concatenate((y_train, y_temp), axis=0)


# Check points
# Showing shapes of the data
print(x_train.shape)
print(y_train.shape)
print()


# Slicing 1st zero-valued row
x_train = x_train[1:]
y_train = y_train[1:]


# Check points
# Showing shapes of the data
print(x_train.shape)
print(y_train.shape)
print()


# Reading binary file with images for testing
# Opening file to read it in binary mode by 'rb'
with open('cifar10/test_batch.bin', 'rb') as bytestream:
    # Reading 10 000 images of 32x32 pixels each
    # Every image has its class index at the beginning
    data = bytestream.read(30730000)

    # Check point
    # Showing type of the data
    print(type(data))

    # Converting data into Numpy array
    data = np.frombuffer(data, dtype=np.uint8)

    # Check point
    # Showing type of the data after conversion
    print(type(data))

    # Reshaping data to 10 000 separate rows
    data = data.reshape(10000, 3073)

    # Extracting class indexes from every row
    y_test = data[:, 0:1]

    # Reshaping Numpy array with class indexes
    y_test = y_test.reshape(10000)

    # Extracting images of 32x32 pixels each from every row
    x_test = data[:, 1:].astype(np.float32)

    # Reshaping Numpy array with images
    # to make every image as separate matrix of 32x32 pixels and 3 channels
    x_test = x_test.reshape(10000, 3, 32, 32)

    # Making channels come at the end
    x_test = x_test.transpose(0, 2, 3, 1)

    # Check points
    # Showing shapes of the data
    print(x_test.shape)
    print(y_test.shape)
    print()

"""
End of:
Reading binary files of CIFAR-10 dataset
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
# Creating file with name 'dataset_cifar10.hdf5'
# Opening it in writing mode by 'w'
with h5py.File('dataset_cifar10.hdf5', 'w') as f:
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
