#  Data Set & Epochs & Regression Train steps & Estimated Epochs &  Estimation Point & Acc Gap & Loss Gap \\
# MNIST & 21 & 2 & 3 & 15 & ? & ? \\

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras import datasets, layers, models
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
sys.path.append(os.getcwd())
from tools.model_info import save_graph_plot, save_graph_json
from tools.project_tools import get_project_name, get_project_paths
from tools.model_info import save_graph_plot, save_graph_json
from CustomCallback.storeWeightsNew import StoreWeights
from tools.create_charts import drawPlot_acc_loss
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback,CSVLogger
from tensorflow.keras import models, layers, losses

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import shutil

model_list = []
model_name_list = []

epochs = 20

skip_steps = [
    [15,3]
]
#check if totalskips + last skip at is within total epochs
if sum(j for i,j in skip_steps) + max(skip_steps)[0] > epochs:
    print("Exiting:skip_steps + last skip exceeds total Epochs")
    quit()

FirstSkip = skip_steps[0][0]

TotalSkips = sum([j for i,j in skip_steps])

reg_train_steps = 2
clusters = 5
project_paths = get_project_paths(sys.argv[0], to_tmp=False)

callback_weights_reg = StoreWeights(project_paths["weights"], reg_train_steps=0,dtw_clusters=0, file_prefix ="Oweights" , weight_pred_ind=False,weighs_dtw_cluster_ind=False, replicate_csvs_at = FirstSkip )

callback_weights_pred = StoreWeights(project_paths["weights"], reg_train_steps= reg_train_steps,dtw_clusters=0, file_prefix ="Rweights" ,skip_array=skip_steps, weight_pred_ind=True,weighs_dtw_cluster_ind=True, replicate_csvs_at = 0)



num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
train_images = train_images.astype("float32") / 255
test_images = test_images.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
train_images = np.expand_dims(train_images, -1)
test_images = np.expand_dims(test_images, -1)

# convert class vectors to binary class matrices
train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)

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

logs = project_paths["weights"] + "/reg_model_history_log.csv"
csv_logger = CSVLogger(logs, append=True)

shutil.copy2(os.path.realpath(__file__), project_paths['code'])  # copying code file to logs

checkpoint_path = project_paths["checkpoints"] + "/weights_epoch-{epoch}.ckpt"
restore_path = project_paths["checkpoints"] + "/weights_epoch-" + str(FirstSkip) + ".ckpt"

callback_save_model_reg= ModelCheckpoint(filepath=checkpoint_path, save_weights_only=False,
                                         verbose=1, period = FirstSkip)

#                             weight_pred_ind=1,
#                             skip_info=skip_info
model = create_model()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
save_graph_plot(model, project_paths["plots"] + "/reg_model.ps")
save_graph_json(model, project_paths["plots"] + "/reg_model.json")

reg_model_hist = model.fit(train_images, train_labels, epochs=epochs,
                               validation_data=(test_images, test_labels),
                               callbacks=[csv_logger,callback_save_model_reg,callback_weights_reg])

model_list.append(reg_model_hist)
model_name_list.append("Regular model ")

# Weight estimation model
pred_model = tf.keras.models.load_model(restore_path)

pred_model_hist = pred_model.fit(train_images, train_labels, epochs=epochs - TotalSkips-1 ,
                                 initial_epoch = FirstSkip ,
                                 validation_data=(test_images, test_labels),
                                 callbacks=[csv_logger,callback_weights_pred])

model_list.append(pred_model_hist)
model_name_list.append("Pred Model ")

drawPlot_acc_loss(model_list, model_name_list,project_paths["plots"])
