'''
Run a simple nn model
take weights at certain steps,
run DTW, clusterisation,
feed regression weights to csv and then to network
rerun network
'''

import tensorflow as tf
import sys
import os

sys.path.append(os.getcwd())
from tools.model_info import save_graph_plot, save_graph_json
from tools.project_tools import get_project_name, get_project_paths
from tools.model_info import save_graph_plot, save_graph_json
from CustomCallback.StoringWeights import StoreWeights
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback,CSVLogger

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import shutil

model_list = []
model_name_list = []
epochs = 100
model_name = "Reg_Model"
skip_steps = [
    [4, 1],
    [9, 1],
    [15, 1],
    [21, 1],
    [27, 1],
    [33, 1],
    [41, 1],
    [50, 1],
    [62, 1],
    [74, 1],
    [86, 1]
]

#check if totalskips + last skip at is within total epochs
total_skips = sum(j for i,j in skip_steps)
if sum(j for i,j in skip_steps) + max(skip_steps)[0] > epochs:
    print("Exiting:skip_steps + last skip exceeds total Epochs")
    quit()

reg_train_steps = 3
clusters = 5
project_paths = get_project_paths(sys.argv[0], to_tmp=False)
callback_weights = StoreWeights(project_paths["weights"], reg_train_steps=reg_train_steps, dtw_clusters=clusters,skip_array=skip_steps, weight_pred_ind=True, weighs_dtw_cluster_ind=True)



fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0


# trimming inputs for fast processing
train_images = train_images[0:500,5:15,5:15]
train_labels = train_labels[0:500]
test_images = test_images[0:50,5:15,5:15]
test_labels = test_labels[0:50]


shutil.copy2(os.path.realpath(__file__), project_paths['code'])  # copying code file to logs
checkpoint_path = "training_1/cp.ckpt"  # redefine
checkpoint_dir = os.path.dirname(checkpoint_path)



# Weights mechanism
# skip_steps [skip at epoch number, skip these number of steps]

reg_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(10, 10)),
    tf.keras.layers.Dense(15, activation='relu'),
    tf.keras.layers.Dense(10)
])

logs = project_paths["logs"] + "/" + "regModelE" + str(epochs) + "Skp"+ total_skips .csv"
csv_logger = CSVLogger(logs, append=True)

shutil.copy2(os.path.realpath(__file__), project_paths['code'])  # copying code file to logs


#storing model architecture

save_graph_plot(reg_model, project_paths["plots"] + "/reg_model.ps")
save_graph_json(reg_model, project_paths["plots"] + "/reg_model.json")

reg_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
reg_model.fit(train_images, train_labels, epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[csv_logger,callback_weights])
