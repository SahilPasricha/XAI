

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import sys
import os
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

epochs = 10

skip_steps = [
    [5,1]
]
#check if totalskips + last skip at is within total epochs
if sum(j for i,j in skip_steps) + max(skip_steps)[0] > epochs:
    print("Exiting:skip_steps + last skip exceeds total Epochs")
    quit()

FirstSkip = skip_steps[0][0]

TotalSkips = sum([j for i,j in skip_steps])

reg_train_steps = 1
clusters = 5
project_paths = get_project_paths(sys.argv[0], to_tmp=False)

callback_weights_reg = StoreWeights(project_paths["weights"], reg_train_steps=0,dtw_clusters=0, file_prefix ="Oweights" , weight_pred_ind=False,weighs_dtw_cluster_ind=False, replicate_csvs_at = FirstSkip )

callback_weights_pred = StoreWeights(project_paths["weights"], reg_train_steps= reg_train_steps,dtw_clusters=0, file_prefix ="Rweights" ,skip_array=skip_steps, weight_pred_ind=True,weighs_dtw_cluster_ind=True, replicate_csvs_at = 0)


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


logs = project_paths["weights"] + "/reg_model_history_log.csv"
csv_logger = CSVLogger(logs, append=True)

shutil.copy2(os.path.realpath(__file__), project_paths['code'])  # copying code file to logs

checkpoint_path = project_paths["checkpoints"] + "/weights_epoch-{epoch}.ckpt"
restore_path = project_paths["checkpoints"] + "/weights_epoch-" + str(FirstSkip) + ".ckpt"

callback_save_model_reg= ModelCheckpoint(filepath=checkpoint_path, save_weights_only=False,
                                         verbose=1, period = FirstSkip)

#                             weight_pred_ind=1,
#                             skip_info=skip_info

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10))
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])




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
