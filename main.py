import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

data = pd.read_csv(r"E:\a-z alphabets\A_Z Handwritten Data.csv").astype('float32')
print(data.head(10))
<div class="open_grepper_editor" title="Edit & Save To Grepper"></div>