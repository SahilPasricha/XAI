"""
Title: Simple MNIST convnet
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2015/06/19
Last modified: 2020/04/21
Description: A simple convnet that achieves ~99% test accuracy on MNIST.
"""

"""
## Setup
"""

from tensorflow import keras
from tensorflow.keras import layers

# Build MNIST to late est Pred on it . thjis being simpler main revela clear impacts of pred on it 

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback,CSVLogger
import sys
import os
import datetime
import numpy as np
sys.path.append(os.getcwd())
from tools.project_tools import get_project_name, get_project_paths
from tools.model_info import save_graph_plot, save_graph_json


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
epochs = 11
batch_size = 50
validation_split = 0.2
skip_epoch = 3
skip_from = 5

tfds.disable_progress_bar()
tf.enable_v2_behavior()
project_paths = get_project_paths(sys.argv[0], to_tmp=False)
logs = project_paths["weights"] + "/reg_model_history_log.csv"
csv_logger = CSVLogger(logs, append=True)
(train_images, train_labels), (test_images, test_labels) =  tf.keras.datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0



"""
## Prepare the data
"""

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
## Build the model
"""
def create_model():
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(2, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(4, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model



model1 = create_model()
model1.summary()
save_graph_plot(model1, project_paths["plots"] + "/reg_model.ps")
save_graph_json(model1, project_paths["weights"] + "/reg_model.json")
date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = project_paths["tb"] + "/reg/" + date_time
tensorboard_callback1 = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
checkpoint_path = project_paths["checkpoints"] + "/weights_epoch-{epoch}.ckpt" 
checkpoint_dir = os.path.dirname(checkpoint_path)



"""
## Train the model
"""

batch_size = 128


model1.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history1 = model1.fit(x_train, y_train, 
                    steps_per_epoch=x_train.shape[0] / batch_size  ,  # 5 epochs per full dataset rotation
                    epochs = (epochs - skip_epoch), 
                    batch_size=batch_size,
                    #validation_split=validation_split,
                    validation_data=(x_test, y_test),
					callbacks=[
						ModelCheckpoint(
							filepath=checkpoint_path, 
                            save_weights_only=False,
                            period = 1,
                            weight_pred_ind=1,
                            skip_from = skip_from,
                            skip_epoch = skip_epoch),
						csv_logger,
                        tensorboard_callback1
                        ]
					)

test_loss, test_acc = model1.evaluate(x_test,  y_test, verbose=2)

print("Test loss:", test_loss)
print("Test accuracy:", test_acc)




print ("<-------------------Reinitializing the Orig model ----------------->")
logs2 = project_paths["weights"] + "/orig_model_history_log.csv"
csv_logger2 = CSVLogger(logs2, append=True)
load_model = project_paths["checkpoints"] + "/weights_epoch-"+ str(skip_from) + ".ckpt"
print("Loading model")
model2 =  tf.keras.models.load_model(load_model)
save_graph_plot(model2, project_paths["plots"] + "/orig_model.ps")
save_graph_json(model2, project_paths["weights"] + "/orig_model.json")
log_dir2 = project_paths["tb"] + "/orig/" + date_time
tensorboard_callback2 = tf.keras.callbacks.TensorBoard(log_dir=log_dir2, histogram_freq=1)
history2 = model2.fit(x_train, y_train, 
                    steps_per_epoch=x_train.shape[0] / batch_size  ,  # 5 epochs per full dataset rotation
                    epochs=epochs-1 ,
                    initial_epoch = skip_from-1,
                    batch_size=batch_size,
                    #validation_split=validation_split,
                    validation_data=(x_test, y_test),
					callbacks=[
						ModelCheckpoint(
							project_paths["checkpoints"] + "/orig_weights_epoch-{epoch:02d}.hdf5",
                            save_weights_only=False,
                            period = 1),
						csv_logger2,
                        tensorboard_callback2
                        ]
					)
test_loss, test_acc = model2.evaluate(x_test, y_test, verbose=2)
print(f'Orig --> Test loss: {test_loss} / Test accuracy: {test_acc}')




