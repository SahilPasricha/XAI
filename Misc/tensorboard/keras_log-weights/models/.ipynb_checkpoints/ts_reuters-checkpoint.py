from __future__ import print_function


from tensorflow.keras.datasets import reuters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback,CSVLogger

import tensorflow as tf
import numpy as np
import sys
import os
import keras

sys.path.append(os.getcwd())


from tools.project_tools import get_project_name, get_project_paths
from tools.model_info import save_graph_plot, save_graph_json


max_words = 500
batch_size = 32
epochs = 151

print('Loading data...')
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words,test_split=0.2)
#import pdb;pdb.set_trace()
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

num_classes = np.max(y_train) + 1
print(num_classes, 'classes')

print('Vectorizing sequence data...')
tokenizer = Tokenizer(num_words=max_words)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Convert class vector to binary class matrix '
      '(for use with categorical_crossentropy)')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

print('Building model...')
model = Sequential()

model.add(Dense(5, input_shape=(max_words,)))
model.add(Activation('relu'))
#model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(512))
#model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


project_paths = get_project_paths(sys.argv[0], to_tmp=False)
save_graph_plot(model, project_paths["plots"] + "/model.ps")
save_graph_json(model, project_paths["weights"] + "/model.json")
weight = model.get_weights()
logs = project_paths["weights"] + "/model_history_log.csv"
csv_logger = CSVLogger(logs, append=True)

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
#                      callbacks=[ModelCheckpoint(
#                          project_paths["checkpoints"] + "/weights_epoch-{epoch:02d}.hdf5",
#                          save_weights_only=True,
#                          save_freq='epoch'),csv_logger],
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])