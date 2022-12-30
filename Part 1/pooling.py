

"""
Course:  Convolutional Neural Networks for Image Classification

Section-1
Quick Win #2: Pooling

Description:
Apply max pooling operation to grayscale image
Demonstrate downsampled output image

File: pooling.py
"""


# Algorithm:
# --> Reading coloured image
# --> Converting image to GRAY
# --> Implementing max pooling operation to GRAY image
#
# Result: Plot with input GRAY image and downsampled image after pooling


# Importing needed libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Importing library to see calculation progress inside loops in Real Time
# To install, use following command: pip install tqdm
# Don't forget to activate environment in which you're working
from tqdm import tqdm


"""
Start of:
Reading input image and converting into GRAY
"""

# Reading coloured input image by OpenCV library
# In this way image is opened already as Numpy array

# (!) OpenCV by default reads image in BGR format (order of channels)
# (!) On Windows, the path might look like following:
# r'images\cat.png'
# or:
# 'images\\cat.png'
image_BGR = cv2.imread('images/cat.png')


# Converting input image to GRAY by OpenCV function
image_GRAY = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2GRAY)


# Check point
# Showing type and shape of loaded coloured image
# Showing shape of GRAY image
print('Type of image_BGR is:', type(image_BGR))
print('Shape of image_BGR is:', image_BGR.shape)
print('Shape of image_GRAY is:', image_GRAY.shape)

"""
End of: 
Reading input image and converting into GRAY
"""


"""
Start of:
Pooling operation to GRAY image
"""

# Preparing hyperparameters for pooling
# To get resulted image after pooling,
# it is needed to set following:
# filter size for pooling (width and height are equal)
f_pooling = 2
# stride (step) for sliding
step = 2

# Output image's dimension is calculated by following equations:
# height_out = (height_in - f_pooling) / step + 1
# width_out = (width_in - f_pooling) / step + 1

# For instance, input GRAY image is 1280x720 of spatial size (width and height),
# then output image after pooling will be as following:
# height_out = (720 - 2) / 2 + 1 = 360
# width_out = (1280 - 2) / 2 + 1 = 640


# Calculating spatial size of output resulted array (width and height)
# Making output width and height as integer numbers,
# in case input width/height is odd
height_out = int((image_GRAY.shape[0] - 2) / 2 + 1)
width_out = int((image_GRAY.shape[1] - 2) / 2 + 1)


# Preparing zero valued output array for image after pooling
image_after_pooling = np.zeros((height_out, width_out))


# Check point
# Showing shape of image after pooling
print('Shape of image after pooling is:', image_after_pooling.shape)


# Implementing pooling operation
# Preparing indexes for rows of output array
ii = 0

# Sliding through entire input GRAY image
# Wrapping the loop with 'tqdm' in order to see progress in real time
for i in tqdm(range(0, image_GRAY.shape[0] - f_pooling + 1, step)):
    # Preparing indexes for columns of output array
    jj = 0

    for j in range(0, image_GRAY.shape[1] - f_pooling + 1, step):
        # Extracting (slicing) a 2x2 patch (the same size with filter)
        # from input GRAY image
        patch = image_GRAY[i:i+f_pooling, j:j+f_pooling]

        # Applying max pooling operation - choosing maximum element
        # from the current patch
        image_after_pooling[ii, jj] = np.max(patch)

        # Increasing indexes for rows of output array
        jj += 1

    # Increasing indexes for columns of output array
    ii += 1


"""
End of:
Pooling operation to GRAY image
"""


"""
Start of:
Plotting resulted image after pooling
"""

# Setting default size of the plot
plt.rcParams['figure.figsize'] = (18.0, 10.0)


# Defining a figure object with number of needed subplots
# ax is a (2,) Numpy array and to access specific subplot we call it by ax[0]
# 'gridspec_kw' is the dictionary with keywords to specify the ratio of subplots
figure, ax = plt.subplots(nrows=1, ncols=2,
                          gridspec_kw={'width_ratios': [2, 1]})


# Adjusting first column with input GRAY image
ax[0].imshow(image_GRAY, cmap=plt.get_cmap('gray'))


# Adjusting second column with GRAY image after pooling
ax[1].imshow(image_after_pooling, cmap=plt.get_cmap('gray'))


# Giving names to columns
ax[0].set_title('Input GRAY', fontsize=16)
ax[1].set_title('After Pooling', fontsize=16)


# Adjusting distance between subplots
plt.tight_layout()
plt.subplots_adjust(left=0.1, right=0.9,
                    bottom=0.1, top=0.9,
                    wspace=0.1, hspace=0.1)


# Saving the plot
# (!) On Windows, the path might look like following:
# r'images\plot_pooling.png'
# or:
# 'images\\plot_pooling.png'
figure.savefig('images/plot_pooling.png')

# Giving name to the window with figure
figure.canvas.set_window_title('Pooling operation to GRAY image')

# Showing the plot
plt.show()

"""
End of:
Plotting resulted image after pooling
"""


"""
Some comments
To get more details of usage subplots from matplotlib library:
print(help(plt.subplots))
print(help(plt.subplots_adjust))


More details and examples are here:
https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.subplots.html
"""
