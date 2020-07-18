# To pkeras_model=None training, we import the necessary functions and submodules from keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dropout, Reshape, Dense, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.optimizers import Adadelta, SGD, RMSprop;
import keras.losses;
from keras.constraints import maxnorm;
from keras.utils import normalize, to_categorical
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.callbacks import EarlyStopping, History, ModelCheckpoint
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, precision_recall_curve, auc

# plt.use('Agg')
from sklearn.model_selection import GridSearchCV, train_test_split

K.set_image_data_format('channels_last')

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


#####################
##Load the data######
#####################
def load_data():
    df = pd.read_csv("/home/yuxuan/dp/eif3a_full_conpositionDen.csv")
    # print(df)
    n = len(df.columns)
    train = int(n / 2)
    x_train = df.iloc[:, 2:train]

    x_val = df.iloc[:, (train + 1):(n - 1)]
    x_val = pd.DataFrame(x_val)
    x_val = x_val.dropna()
    # print(x_val)

    # x_train = np.expand_dims(x_train, axis=1)
    # x_val = np.expand_dims(x_val, axis=1)

    y_train = df.iloc[:, train:train + 1]
    y_val = df.iloc[:, (n - 1):]
    y_val = DataFrame(y_val)
    y_val = y_val.dropna()
    y_val = DataFrame(y_val, dtype=int)

    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5)

    print(x_val.shape)
    print(x_train.shape)
    print(x_test.shape)
    print(y_val.shape)
    print(y_train.shape)
    print(y_test.shape)

    return x_train, x_test, x_val, y_test, y_train, y_val

x_train, x_test, x_val, y_test, y_train, y_val = load_data()
model = Sequential()
model.add(Dense(1200,input_shape=x_train.shape[1::]))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(200))
model.add(Activation("softmax"))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.summary()
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=1e-4),
              metrics=['accuracy'])

epoch = 100
batchsize = 32
earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=10,
                          verbose=1)

callbacks_list = [earlystop]
history = model.fit(x_train, y_train, batch_size=batchsize, epochs=epoch,
                    validation_data=(x_test, y_test),
                    callbacks=callbacks_list)
