#%%
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

import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, precision_recall_curve, auc

#%%
def load_data():
    df = pd.read_csv("eif3a_full_test.csv")

    train_All_1 = df.iloc[:, 2:167]
    # train_All_1
    test_all_1 = df.iloc[0:90, 167:332]
    # test_all_1

    y_train = df.iloc[:,166:167]
    y_test = df.iloc[0:90,331:332]
    x_train = train_All_1.iloc[:,:-1]
    x_test = test_all_1.iloc[:,:-1]

    round = int(len(x_train.columns) / 4)
    lst = []
    for i in range(round):
        x = x_train.iloc[:, (4 * i):(4 * i + 4)]
        lst.append(x)
    ls = []
    for j in range(round):
        x = lst[j]
        y = np.array(x).tolist()
        ls.append(y)
    test = np.array(ls)
    x_train = np.swapaxes(test, axis1=0, axis2=1)

    round2 = int(len(x_test.columns) / 4)
    lst2 = []
    for i in range(round2):
        x = x_test.iloc[:, (4 * i):(4 * i + 4)]
        lst2.append(x)
    ls2 = []
    for j in range(round2):
        x = lst2[j]
        y = np.array(x).tolist()
        ls2.append(y)
    test2 = np.array(ls2)
    x_test = np.swapaxes(test2, axis1=0, axis2=1)

    y_train = np.array([False, True])
    y_train = y_train.repeat(711)
    y_train = np.mat(y_train).transpose()
    # y_train = to_categorical(y_train)
    # print(y_train)

    y_test = np.array([False, True])
    y_test = y_test.repeat(45)
    y_test = np.mat(y_test).transpose()
    # y_test = to_categorical(y_test)

    return x_train, x_test, y_test, y_train
#%%
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

#%%
def build_model(x_train):
    one_filter_keras_model = Sequential()
    one_filter_keras_model.add(
        Conv1D(filters=90, kernel_size=5, padding="valid", kernel_regularizer=regularizers.l2(0.01),
               input_shape=x_train.shape[1::]))
    one_filter_keras_model.add(Activation('relu'))
    one_filter_keras_model.add(MaxPooling1D(pool_size=4, strides=2))
    one_filter_keras_model.add(Dropout(0.25))

    one_filter_keras_model.add(
        Conv1D(filters=100, kernel_size=3, padding="valid", kernel_regularizer=regularizers.l2(0.01)))
    one_filter_keras_model.add(Activation('relu'))
    one_filter_keras_model.add(MaxPooling1D(pool_size=10, strides=1))
    one_filter_keras_model.add(Dropout(0.25))

    one_filter_keras_model.add(Flatten())
    one_filter_keras_model.add(Dense(1421))
    one_filter_keras_model.add(Activation("relu"))
    one_filter_keras_model.add(Dense(1))
    one_filter_keras_model.add(Activation("sigmoid"))
    one_filter_keras_model.summary()

    adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
    one_filter_keras_model.compile(loss='binary_crossentropy', optimizer='adam',
                                   metrics=['accuracy', precision, recall])
    return one_filter_keras_model
#%%
def compileModel(model, x_train, x_test, y_test, y_train):
    model = model
    x_train = x_train
    x_test = x_test
    y_test = y_test
    y_train = y_train
    earlystop = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=10,
                              verbose=1)

    filepath = "weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, earlystop]

    epoch = 100
    batchsize = 128

    history = model.fit(x_train, y_train, batch_size=batchsize, epochs=epoch,
                        validation_data=(x_test, y_test),
                        callbacks=callbacks_list)
    return history

# ################################
# print('draw the loss plot')
# ###############################
def lossplot(history):
    ori_val_Loss = history.history['val_loss']
    loss = history.history['loss']
    epochs = np.arange(len(history.epoch)) + 1
    plt.plot(epochs, ori_val_Loss, label='val loss')
    plt.plot(epochs, loss, label='loss')
    plt.title("Effect of model capacity on validation loss\n")
    plt.xlabel('Epoch #')
    plt.ylabel('Validation Loss')
    plt.legend()
    # plt.show()
    plt.savefig('loss_encoding.png')
    print("")
    print("The loss plot is saved \n")


def roc(model, x_test, y_test):
    print('Start drawing the roc curve \n')
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc
    y_pred_keras = model.predict(x_test).ravel()
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
    auc_keras = auc(fpr_keras, tpr_keras)

    plt.cla()
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='AUROC (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    # plt.show()
    print('AUROC (area = {:.3f})'.format(auc_keras))
    plt.savefig('ROC_encoding.png')
    return auc_keras


def prcurve(model, x_test, y_test):
    lr_probs = model.predict_proba(x_test)
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
    lr_auc = auc(lr_recall, lr_precision)

    # summarize scores
    print('PRAUC:  auc=%.3f' % (lr_auc))
    # plot the precision-recall curves
    no_skill = len(y_test[y_test == 1]) / len(y_test)
    pyplot.cla()
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    pyplot.plot(lr_recall, lr_precision, marker='.', label='Logistic')
    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    # show the legend
    pyplot.legend()
    # show the plot
    # pyplot.show()
    plt.savefig('PRAUC_encoding.png')
    return lr_auc


def main():
    x_train, x_test, y_test, y_train = load_data()
    model = build_model(x_train)
    history = compileModel(model, x_train, x_test, y_test, y_train)
    lossplot(history)
    roc(model, x_test, y_test)
    prcurve(model, x_test, y_test)


if __name__ == '__main__':
    main()





