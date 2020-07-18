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

# plt.use('Agg')

K.set_image_data_format('channels_last')

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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


def seq_to_mat(seq):
    seq_len = len(seq)
    seq = seq.replace('A', '0')
    seq = seq.replace('a', '0')
    seq = seq.replace('C', '1')
    seq = seq.replace('c', '1')
    seq = seq.replace('G', '2')
    seq = seq.replace('g', '2')
    seq = seq.replace('T', '3')
    seq = seq.replace('t', '3')
    seq = seq.replace('U', '3')
    seq = seq.replace('u', '3')
    seq = seq.replace('N', '4')
    seq = seq.replace('n', '4')
    seq_code = np.zeros((4, seq_len), dtype='float16')
    for i in range(seq_len):
        if int(seq[i]) != 4:
            seq_code[int(seq[i]), i] = 1
        else:
            seq_code[0:4, i] = np.tile(0.25, 4)
    return np.transpose(seq_code)


#####################
##Load the data######
#####################
def load_data():
    df = pd.read_csv("eif3a_full_40.csv")

    # print(df)

    train_All_1 = df.iloc[:, 2]
    test_all_1 = df.iloc[:, 3]

    pos_train_seq = train_All_1[0:711, ]
    neg_train_seq = train_All_1[711:, ]
    pos_test_seq = test_all_1[0:45, ]
    neg_test_seq = test_all_1[45:90, ]

    # %%

    X_train = np.array(train_All_1)
    # a = X_train[1]
    # print(len(a))
    lt = []
    for seq in X_train:
        x = seq_to_mat(seq)
        lt.append(x)

    x_train = np.array(lt)
    # print(x_train.shape)

    lst_test = []
    X_test = np.array(test_all_1[0:90])

    for seqs in X_test:
        x = seq_to_mat(seqs)
        lst_test.append(x)

    x_test = np.array(lst_test)

    y_train = np.array([True, False])
    y_train = y_train.repeat(711)
    y_train = np.mat(y_train).transpose()
    # y_train = to_categorical(y_train)
    # print(y_train)

    y_test = np.array([True, False])
    y_test = y_test.repeat(45)
    y_test = np.mat(y_test).transpose()
    # y_test = to_categorical(y_test)

    # print(X_train)
    # print(y_train)
    # print(X_test)
    # print(y_test)

    print("x_train", x_train.shape)
    print('x_test', x_test.shape)
    print('y_test', y_test.shape)
    print('y_train', y_train.shape)
    return x_train, x_test, y_test, y_train


##########################################################
#####Define the model architecture in keras###############
#########################################################

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

    filepath = "weights.best_encoding.hdf5"
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
    plt.cla()
    plt.plot(epochs, ori_val_Loss, label='val loss')
    plt.plot(epochs, loss, label='loss')
    plt.title("Effect of model capacity on validation loss\n")
    plt.xlabel('Epoch #')
    plt.ylabel('Validation Loss')
    plt.legend()
    # plt.show()
    plt.savefig('/home/yuxuan/dp/onehot/lossplot.png')
    print("")
    print("The loss plot is saved \n")

def MCC(model,x_test,y_test):
    from sklearn.metrics import matthews_corrcoef
    yhat = model.predict_classes(x_test)
    mcc = matthews_corrcoef(y_test, yhat)
    print('MCC = {:.3f})'.format(mcc))
    return mcc

def ACC(model,x_test,y_test):
    from sklearn.metrics import accuracy_score
    yhat = model.predict_classes(x_test)
    acc = accuracy_score(y_test, yhat)
    print('ACC = {:.3f})'.format(acc))
    return acc

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
    plt.savefig('/home/yuxuan/dp/onehot/ROC.png')
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
    plt.savefig('/home/yuxuan/dp/onehot/PRAUC.png')
    return lr_auc


def main():
    x_train, x_test, y_test, y_train = load_data()
    model = build_model(x_train)
    history = compileModel(model, x_train, x_test, y_test, y_train)
    lossplot(history)
    auc = roc(model, x_test, y_test)
    prauc = prcurve(model, x_test, y_test)
    mcc = MCC(model, x_test, y_test)
    acc = ACC(model, x_test, y_test)
    results = np.array([auc, prauc, mcc, acc])
    np.savetxt('/home/yuxuan/dp/onehot/eif3a_full_onehot.csv', results, delimiter=',')


if __name__ == '__main__':
    main()
