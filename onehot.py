import numpy as np

def seq_to_mat(seq):
    seq_len = len(seq)
    seq = seq.replace('A','0')
    seq = seq.replace('a','0')
    seq = seq.replace('C','1')
    seq = seq.replace('c','1')
    seq = seq.replace('G','2')
    seq = seq.replace('g','2')
    seq = seq.replace('T','3')
    seq = seq.replace('t','3')
    seq = seq.replace('U','3')
    seq = seq.replace('u','3')
    seq = seq.replace('N','4')
    seq = seq.replace('n','4')
    seq_code = np.zeros((4,seq_len), dtype='float16')
    for i in range(seq_len):
        if int(seq[i]) != 4:
            seq_code[int(seq[i]),i] = 1
        else:
            seq_code[0:4,i] = np.tile(0.25,4)
    return np.transpose(seq_code)

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
