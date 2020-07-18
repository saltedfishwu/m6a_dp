#%%
import numpy as np
from keras import models
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification

#%%
# 为了可以重复出我们的结果
np.random.seed(1)
# 产生自己的数据集: 2000个样本点， 50个特征
number_of_features = 50
features, target = make_classification(n_samples = 2000,
                                       n_features = number_of_features,
                                       n_informative = 3,
                                       n_redundant = 0,
                                       n_classes = 2,
                                       weights = [.5, .5],
                                       random_state = 1)


#%%
print(features)
print(target)


#%%

def create_model_DNN(optimizer='rmsprop'):
    model_DNN = models.Sequential()
    model_DNN.add(layers.Dense(units=32, activation='relu', input_shape=(number_of_features,)))
    model_DNN.add(layers.Dense(units=32, activation='relu'))
    model_DNN.add(layers.Dense(units=1, activation='sigmoid'))
    model_DNN.compile(loss='binary_crossentropy', # Cross-entropy
                    optimizer=optimizer, # Optimizer
                    metrics=['accuracy']) # Accuracy performance metric
    # Return compiled model_DNN
    return model_DNN

#%%
    model_DNN = KerasClassifier(build_fn=create_model_DNN, verbose=1)

#%%

epochs = [5, 20]
batches = [5, 20, 100]
optimizers = ['rmsprop', 'adam']
hyperparameters = dict(optimizer=optimizers, epochs=epochs, batch_size=batches)


#%%
# grid = GridSearchCV(estimator=model_DNN, cv=5, param_grid=hyperparameters)
# grid_result = grid.fit(features, target)
# grid_result.best_params_

#%%



