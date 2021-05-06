from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback,CSVLogger

import tensorflow as tf
import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from tools.project_tools import get_project_name, get_project_paths
from tools.model_info import save_graph_plot, save_graph_json
from sklearn.model_selection import train_test_split

def mnist_dense_model(input_shape=(28, 28), n_classes=10, dropout=0.2, model_name="model"):
    
    
    x_in = Input(shape=input_shape, name="input")

    x_in_flat = Flatten()(x_in)
    x = Dense(5, activation=tf.nn.leaky_relu, name="hidden_1")(x_in_flat)
    x = Dropout(dropout, name="dropout_1")(x)

    #x = Dense(50, activation=tf.nn.leaky_relu, name="hidden_2")(x)
    #x = Dropout(dropout, name="dropout_2")(x)

    x_out = Dense(n_classes, activation='softmax', name="output")(x)
    return Model(x_in, x_out, name=model_name)


def main(argv):
    
    batch_size = 50
    epochs = 300 
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
   
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    train_size = 0.5
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size)

    # Convert images from [0,255] to [0,1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Extract input dimension and distinct output classes from dataset
    input_shape = x_train.shape[1:]
    num_classes = len(np.unique(y_train))
    num_train_samples = x_train.shape[0]

    # Convert y to categorical one-hot vectors
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    # Create and compile model
    model = mnist_dense_model(input_shape=input_shape,
                              n_classes=num_classes,
                              dropout=0.5,
                              model_name=get_project_name(argv[0]))

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(),
                  metrics=['accuracy'])

    # Print summary and save model as plot and node-link-graph
    project_paths = get_project_paths(argv[0], to_tmp=False)

    save_graph_plot(model, project_paths["plots"] + "/model.ps")
    save_graph_json(model, project_paths["weights"] + "/model.json")
    
    weight = model.get_weights()
    
    logs = project_paths["weights"] + "/model_history_log.csv"
    np.savetxt('weight.csv' , weight , fmt='%s', delimiter=',')
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
