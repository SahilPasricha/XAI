#Motivation - to reduce input size so that network become complicated and need complicated network to solve task

from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras import initializers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback,CSVLogger
from tensorflow import keras
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras import layers 


import tensorflow as tf
import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from tools.project_tools import get_project_name, get_project_paths
from tools.model_info import save_graph_plot, save_graph_json
num_classes = 0

def dense_model1(input_shape, n_classes, dropout, model_name):
    model = keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.summary()
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes))
    return (model)

    #x_in = Input(shape=input_shape, name="input")
    #x_in_flat = Flatten()(x_in)
    #x = Dense(8, activation=tf.nn.leaky_relu, name="hidden_1")(x_in_flat)
    #x = Dropout(dropout, name="dropout_1")(x)
    #x = Dense(10, activation=tf.nn.leaky_relu, name="hidden_2")(x)
    #x = Dropout(dropout, name="dropout_2")(x)
    #x_out = Dense(n_classes, activation='softmax', name="output")(x)
    #return Model(x_in, x_out, name=model_name)
    
    
def main(argv):
    
    batch_size = 500
    epochs =21
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
    c1 = np.concatenate((x_train, x_test), axis=0)
    c2 = np.concatenate((y_train, y_test), axis=0)
    x_train = c1[:-5000]
    y_train = c2[:-5000]
    x_test =  c1[-5000:]
    y_test =  c2[-5000:]
    num_classes = len(np.unique(y_train))
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    input_shape = x_train.shape[1:]
    num_classes = len(np.unique(y_train))
    num_train_samples = x_train.shape[0]

    # Convert y to categorical one-hot vectors
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    # Create and compile model
    
              
    model = dense_model1(input_shape=input_shape,
                              n_classes=num_classes,
                              dropout=0.5,
                              model_name=get_project_name(argv[0]))

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    # Print summary and save model as plot and node-link-graph
    project_paths = get_project_paths(argv[0], to_tmp=False)
    import pdb;pdb.set_trace()
    #save_graph_plot(model, project_paths["plots"] + "/model.ps")
    #save_graph_json(model, project_paths["weights"] + "/model.json")
    
    weight = model.get_weights()
    
    logs = project_paths["weights"] + "/mod_hist_U5_mnist_acc.csv"
    #np.savetxt('weight.csv' , weight , fmt='%s', delimiter=',')
    csv_logger = CSVLogger(logs, append=True)
    
    
    # Train model while saving weights as checkpoints after each epoch
    model.fit(x_train, y_train,
              steps_per_epoch=num_train_samples / batch_size / 5,  # 5 epochs per full dataset rotation
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
               callbacks=[ModelCheckpoint(
                   project_paths["checkpoints"] + "/weights_epoch-{epoch:02d}.hdf5",
                   save_weights_only=True,
                   save_freq='epoch'),csv_logger],
               validation_data=(x_test, y_test))



if __name__ == "__main__":
    main(sys.argv)
