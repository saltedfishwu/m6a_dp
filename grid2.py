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
    df = pd.read_csv("/home/yuxuan/dp/eif3a_full_m6aReader2.csv")
    # print(df)
    n = len(df.columns)
    train = int(n / 2)
    x_train = df.iloc[:, 2:train]

    x_val = df.iloc[:, (train + 1):(n - 1)]
    x_val = pd.DataFrame(x_val)
    x_val = x_val.dropna()
    # print(x_val)

    x_train = np.expand_dims(x_train, axis=1)
    x_val = np.expand_dims(x_val, axis=1)

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

def build_model(input_shape=[1, 121], optimizer='adam', init_mode='uniform', dropout_rate=0.0, weight_constraint=0):
    one_filter_keras_model = Sequential()
    one_filter_keras_model.add(
        Conv1D(filters=90, kernel_size=1, padding="valid",
               kernel_initializer=init_mode,
               kernel_regularizer=regularizers.l2(0.01),
               kernel_constraint=maxnorm(weight_constraint),
               input_shape=input_shape))
    one_filter_keras_model.add(Activation('relu'))
    one_filter_keras_model.add(MaxPooling1D(pool_size=1, strides=1))
    one_filter_keras_model.add(Dropout(dropout_rate))
    # 注意哈，目前来说只是一层层的试，还没有涉及到两层，到时候一定要都包含进去

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
    x_train, x_test, x_val, y_test, y_train, y_val = load_data()
    # print(x_train.shape[1::])
    model = KerasClassifier(build_fn=build_model, input_shape=x_train.shape[1::], verbose=0)
    # 定义网格搜索参数
    batch_size = [16, 32, 64, 128, 256]
    # batch_size = [32]
    epochs = [50]
    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    # optimizer = ['Adam']
    # init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal',
    #              'glorot_uniform', 'he_normal', 'he_uniform']
    init_mode = ['normal']
    weight_constraint = [5]
    dropout_rate = [0.8] ## 建议最好是从0。1-0。9都包括

    param_grid = dict(batch_size=batch_size,
                      epochs=epochs,
                      optimizer=optimizer,
                      init_mode=init_mode,
                      weight_constraint=weight_constraint,
                      dropout_rate=dropout_rate)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)  ##那个n-jobs 和 pre-dispatch暂时搞不定
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
