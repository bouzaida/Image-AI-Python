

"""
Course:  Convolutional Neural Networks for Image Classification

Section-6
Test Classification in Real Time by camera

Description:
Classify new images by camera in Real Time
Plot bar chart with scores in Real Time

File: test_mnist_model_in_real_time.py
"""


# Algorithm:
# --> Setting up full paths
# --> Loading saved model
# --> Loading and assigning best weights
# --> Loading saved Mean Image and Standard Deviation
# --> Preparing function to plot bar chart
# --> Preparing OpenCV windows to be shown
# --> Reading frames from camera in the loop
#     --> Preprocessing caught frame
#     --> Implementing forward pass
#     --> Showing OpenCV windows with results
#     --> Calculating FPS rate
#
# Result: OpenCV windows with classification results


# Importing needed libraries
import matplotlib.pyplot as plt
import numpy as np
import h5py
import cv2
import io

from keras.models import load_model

from timeit import default_timer as timer


"""
Start of:
Setting up full paths
"""

# Full or absolute path to 'Section4' with preprocessed datasets
# (!) On Windows, the path should look like following:
# r'C:\Users\your_name\PycharmProjects\CNNCourse\Section4'
# or:
# 'C:\\Users\\your_name\\PycharmProjects\\CNNCourse\\Section4'
full_path_to_Section4 = \
    '/home/valentyn/PycharmProjects/CNNCourse/Section4'

# Full or absolute path to 'Section5' with designed models
# (!) On Windows, the path should look like following:
# r'C:\Users\your_name\PycharmProjects\CNNCourse\Section5'
# or:
# 'C:\\Users\\your_name\\PycharmProjects\\CNNCourse\\Section5'
full_path_to_Section5 = \
    '/home/valentyn/PycharmProjects/CNNCourse/Section5'

"""
End of:
Setting up full paths
"""


"""
Start of:
Loading saved model
"""

# Loading model
# (!) On Windows, it might need to change
# this: + '/' +
# to this: + '\' +
# or to this: + '\\' +
model = load_model(full_path_to_Section5 + '/' +
                   'mnist' + '/' +
                   'model_1_mnist_gray.h5')

# Check point
print('Model is successfully loaded')

"""
End of:
Loading saved model
"""


"""
Start of:
Loading and assigning best weights
"""

# Loading and assigning best weights
# (!) On Windows, it might need to change
# this: + '/' +
# to this: + '\' +
# or to this: + '\\' +
model.load_weights('mnist' + '/' + 'w_1_mnist_gray_255_mean_std.h5')

# Check point
print('Best weights are loaded and assigned')

"""
End of:
Loading and assigning best weights
"""


"""
Start of:
Loading saved Mean Image and Standard Deviation
"""

# Opening saved Mean Image for GRAY MNIST dataset
# Initiating File object
# Opening file in reading mode by 'r'
# (!) On Windows, it might need to change
# this: + '/' +
# to this: + '\' +
# or to this: + '\\' +
with h5py.File(full_path_to_Section4 + '/' +
               'mnist' + '/' +
               'mean_gray_dataset_mnist.hdf5', 'r') as f:
    # Extracting saved array for Mean Image
    # Saving it into new variable
    mean_gray = f['mean']  # HDF5 dataset
    # Converting it into Numpy array
    mean_gray = np.array(mean_gray)  # Numpy arrays


# Opening saved Standard Deviation for GRAY MNIST dataset
# Initiating File object
# Opening file in reading mode by 'r'
# (!) On Windows, it might need to change
# this: + '/' +
# to this: + '\' +
# or to this: + '\\' +
with h5py.File(full_path_to_Section4 + '/' +
               'mnist' + '/' +
               'std_gray_dataset_mnist.hdf5', 'r') as f:
    # Extracting saved array for Standard Deviation
    # Saving it into new variable
    std_gray = f['std']  # HDF5 dataset
    # Converting it into Numpy array
    std_gray = np.array(std_gray)  # Numpy arrays

"""
End of:
Loading saved Mean Image and Standard Deviation
"""


"""
Start of:
Preparing function to plot bar chart
"""


# Defining function to plot bar chart with scores values
def bar_chart(obtained_scores):
    # Arranging X axis
    x_positions = np.arange(obtained_scores.size)

    # Creating bar chart
    bars = plt.bar(x_positions, obtained_scores, align='center', alpha=0.6)

    # Highlighting the highest bar
    bars[np.argmax(obtained_scores)].set_color('red')

    # Giving names to axes
    plt.xlabel('Class', fontsize=20)
    plt.ylabel('Value', fontsize=20)

    # Giving name to bar chart
    plt.title('Obtained Scores', fontsize=20)

    # Adjusting borders of the plot
    plt.tight_layout(pad=2.5)

    # Initializing object of the buffer
    b = io.BytesIO()

    # Saving bar chart into the buffer
    plt.savefig(b, format='png', dpi=200)

    # Closing plot with bar chart
    plt.close()

    # Moving pointer to the beginning of the buffer
    b.seek(0)

    # Reading bar chart from the buffer
    bar_image = np.frombuffer(b.getvalue(), dtype=np.uint8)

    # Closing buffer
    b.close()

    # Decoding buffer
    bar_image = cv2.imdecode(bar_image, 1)

    # Returning Numpy array with bar chart
    return bar_image


# Check point
print('Function to plot Bar Chart is successfully defined')


"""
End of:
Preparing function to plot bar chart
"""


"""
Start of:
Preparing OpenCV windows to be shown
"""

# Giving names to the windows
# Specifying that windows are resizable

# Window to show current view from camera in Real Time
cv2.namedWindow('Current view', cv2.WINDOW_NORMAL)

# Window to show classification result
cv2.namedWindow('Classified as', cv2.WINDOW_NORMAL)

# Window to show bar chart with scores
cv2.namedWindow('Scores', cv2.WINDOW_NORMAL)

# Check point
print('OpenCV windows are ready')

"""
End of:
Preparing OpenCV windows to be shown
"""


"""
Start of:
Reading frames from camera in the loop
"""

# Defining 'VideoCapture' object
# to read stream video from camera
# Index of the built-in camera is usually 0
# Try to select other cameras by passing 1, 2, 3, etc.
camera = cv2.VideoCapture(0)


# Defining counter for FPS (Frames Per Second)
counter = 0

# Starting timer for FPS
# Getting current time point in seconds
fps_start = timer()


# Creating image with black background
temp = np.zeros((720, 1280, 3), np.uint8)


# Defining loop to catch frames
while True:
    # Capturing frames one-by-one from camera
    _, frame_bgr = camera.read()

    """
    Start of:
    Preprocessing caught frame
    """

    # Converting frame to GRAY by OpenCV function
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Resizing frame to 28 by 28 pixels size
    frame_gray = cv2.resize(frame_gray,
                            (28, 28),
                            interpolation=cv2.INTER_CUBIC)

    # Extending dimension from (height, width) to (height, width, 1)
    frame_gray = frame_gray[:, :, np.newaxis]

    # Implementing normalization by dividing image's pixels on 255.0
    frame_gray_255 = frame_gray / 255.0

    # Implementing normalization by subtracting Mean Image
    frame_gray_255_mean = frame_gray_255 - mean_gray

    # Implementing preprocessing by dividing on Standard Deviation
    frame_gray_255_mean_std = frame_gray_255_mean / std_gray

    # Extending dimension from (height, width, 1)
    # to (1, height, width, 1)
    frame_gray_255_mean_std = frame_gray_255_mean_std[np.newaxis, :, :, :]

    """
    End of:
    Preprocessing caught frame
    """

    """
    Start of:
    Implementing forward pass
    """

    # Testing GRAY MNIST model trained on dataset:
    # dataset_mnist_gray_255_mean_std.hdf5
    # Caught frame is preprocessed in the same way
    # Measuring classification time
    start = timer()
    scores = model.predict(frame_gray_255_mean_std)
    end = timer()

    # Scores are given as 10 numbers of predictions for each class
    # Getting index of only one class with maximum value
    prediction = np.argmax(scores)

    """
    End of:
    Implementing forward pass
    """

    """
    Start of:
    Showing OpenCV windows
    """

    # Showing current view from camera in Real Time
    # Pay attention! 'cv2.imshow' takes images in BGR format
    cv2.imshow('Current view', frame_bgr)

    # Changing background to BGR(230, 161, 0)
    # B = 230, G = 161, R = 0
    temp[:, :, 0] = 230
    temp[:, :, 1] = 161
    temp[:, :, 2] = 0

    # Adding text with current label
    cv2.putText(temp, str(prediction), (100, 200),
                cv2.FONT_HERSHEY_TRIPLEX, 6, (255, 255, 255), 6, cv2.LINE_AA)

    # Adding text with obtained confidence score to image with label
    cv2.putText(temp, 'Score : ' + '{0:.5f}'.format(scores[0][prediction]),
                (100, 450), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 255, 255),
                4, cv2.LINE_AA)

    # Adding text with time spent for classification to image with label
    cv2.putText(temp, 'Time  : ' + '{0:.5f}'.format(end - start),
                (100, 600), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 255, 255),
                4, cv2.LINE_AA)

    # Showing image with respect to classification results
    cv2.imshow('Classified as', temp)

    # Showing bar chart
    cv2.imshow('Scores', bar_chart(scores[0]))

    """
    End of:
    Showing OpenCV windows
    """

    """
    Start of:
    Calculating FPS
    """

    # Increasing counter for FPS
    counter += 1

    # Stopping timer for FPS
    # Getting current time point in seconds
    fps_stop = timer()

    # Checking if timer reached 1 second
    # Comparing
    if fps_stop - fps_start >= 1.0:
        # Showing FPS rate
        print('FPS rate is: ', counter)

        # Reset FPS counter
        counter = 0

        # Restart timer for FPS
        # Getting current time point in seconds
        fps_start = timer()

    """
    End of:
    Calculating FPS
    """

    # Breaking the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

"""
End of:
Reading frames from camera in the loop
"""


# Releasing camera
camera.release()

# Destroying all opened OpenCV windows
cv2.destroyAllWindows()


"""
Some comments
'io.BytesIO' object is an in-memory binary stream.
More details and examples are here:
print(help(io))
print(help(io.BytesIO()))
https://docs.python.org/3.7/library/io.html


Function 'np.frombuffer' interprets a buffer as a 1-dimensional array.
More details and examples are here:
print(help(np.frombuffer))
https://numpy.org/doc/stable/reference/generated/numpy.frombuffer.html


Function 'cv2.imdecode' reads image from a buffer.
More details are here:
print(help(cv2.imdecode))


Function 'tight_layout' adjusts padding of the plot.
More details are here:
print(help(plt.tight_layout))
https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.tight_layout.html


Function 'cv2.putText' adds text to images.
More details and examples are here:
print(help(cv2.putText))
https://docs.opencv.org/master/dc/da5/tutorial_py_drawing_functions.html


To set height and width of the captured frames add following lines
right after VideoCapture object:

camera = cv2.VideoCapture(0)
camera.set(3, 1280)
camera.set(4, 720)

equivalent is:

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

More details and examples are here:
print(help(cv2.VideoCapture.set))
https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html

Pay attention! This properties might work or not.
The property value has to be accepted by the capture device (camera).

"""
