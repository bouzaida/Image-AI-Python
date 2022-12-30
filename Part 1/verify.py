

"""
Course:  Convolutional Neural Networks for Image Classification

Section-1
Software Installation & Verification

Description:
Install needed prerequisites to the course
Verify successful installation

File: verify.py
"""


# Algorithm:
# --> Run the code
#
# Result: Versions of installed libraries


# Importing libraries
import keras
import cv2
import sklearn
import matplotlib
import pandas
import tqdm
import pydot
import numpy
import h5py


# Printing their versions
print('Keras v:', keras.__version__)
print('OpenCV v:', cv2.__version__)
print('Scikit-learn v:', sklearn.__version__)
print('Matplotlib v:', matplotlib.__version__)
print('Pandas v:', pandas.__version__)
print('Tqdm v:', tqdm.__version__)
print('Pydot v:', pydot.__version__)
print('Numpy v:', numpy.__version__)
print('h5py v:', h5py.__version__)
