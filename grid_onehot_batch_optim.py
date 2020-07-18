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

import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, precision_recall_curve, auc

# plt.use('Agg')
from sklearn.model_selection import GridSearchCV

K.set_image_data_format('channels_last')

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


#####################
##Load the data######
#####################
def load_data():
    df = pd.read_csv("/home/yuxuan/dp/eif3a_full_m6aReader.csv")
    # print(df)
    n = len(df.columns)
    train = int(n / 2)
    x_train = df.iloc[:, 2:train]

    x_test = df.iloc[:, (train + 1):(n - 1)]
    x_test = pd.DataFrame(x_test)
    x_test = x_test.dropna()
    # print(x_test)

    x_train = np.expand_dims(x_train, axis=1)
    x_test = np.expand_dims(x_test, axis=1)

    y_train = np.array([1, 0])
    y_train = y_train.repeat(int((df.shape[0]) / 2))
    y_train = np.mat(y_train).transpose()

    y_test = np.array([1, 0])
    y_test = y_test.repeat(int((x_test.shape[0] / 2)))
    y_test = np.mat(y_test).transpose()

    # print(x_test.shape)
    # print(x_train.shape)
    # print(y_test.shape)
    # print(y_train.shape)

    return x_train, x_test, y_test, y_train


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


##########################################################
#####Define the model architecture in keras###############
#########################################################

def build_model(input_shape=[1, 121], optimizer='adam'):
    one_filter_keras_model = Sequential()
    one_filter_keras_model.add(
        Conv1D(filters=90, kernel_size=1, padding="valid", kernel_regularizer=regularizers.l2(0.01),
               input_shape=input_shape))
    one_filter_keras_model.add(Activation('relu'))
    one_filter_keras_model.add(MaxPooling1D(pool_size=1, strides=1))
    one_filter_keras_model.add(Dropout(0.25))

    one_filter_keras_model.add(
        Conv1D(filters=100, kernel_size=1, padding="valid", kernel_regularizer=regularizers.l2(0.01)))
    one_filter_keras_model.add(Activation('relu'))
    one_filter_keras_model.add(MaxPooling1D(pool_size=1, strides=1))
    one_filter_keras_model.add(Dropout(0.25))

    one_filter_keras_model.add(Flatten())
    one_filter_keras_model.add(Dense(1210))
    one_filter_keras_model.add(Activation("relu"))
    one_filter_keras_model.add(Dense(1))
    one_filter_keras_model.add(Activation("sigmoid"))
    # one_filter_keras_model.summary()

    # 找learning rate之类的活好像不可以一起做
    # optimizer = keras.optimizers.Adam(lr=learn_rate,momentum=momentum)

    one_filter_keras_model.compile(loss='binary_crossentropy', optimizer=optimizer,
                                   metrics=['accuracy', precision, recall])
    return one_filter_keras_model


def main():
    x_train, x_test, y_test, y_train = load_data()
    # print(x_train.shape[1::])
    model = KerasClassifier(build_fn=build_model, input_shape=x_train.shape[1::], verbose=0)
    # 定义网格搜索参数
    # batch_size = [16, 32, 64, 128, 256]
    batch_size = [128, 256]
    epochs = [50]
    # optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    optimizer = ['SGD', 'Adam']
    param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_result = grid.fit(x_train, y_train)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


if __name__ == '__main__':
    main()
