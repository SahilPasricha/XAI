'''
Goal --> make a fuciton that takes in model name
and draws acc and loss of all functions on one chart
'''



import tensorflow as tf
import sys
import os

sys.path.append(os.getcwd())
from tools.model_info import save_graph_plot, save_graph_json
from tools.project_tools import get_project_name, get_project_paths
from tools.model_info import save_graph_plot, save_graph_json
from CustomCallback.StoringWeights import StoreWeights
from tools.create_charts import drawPlot_acc_loss
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback,CSVLogger

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import shutil

model_list = []
model_name_list = []
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


checkpoint_path = "training_1/cp.ckpt"  # redefine
checkpoint_dir = os.path.dirname(checkpoint_path)



# Weights mechanism
# skip_steps [skip at epoch number, skip these number of steps]

reg_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(10, 10)),
    tf.keras.layers.Dense(15, activation='relu'),
    tf.keras.layers.Dense(10)
])

reg_model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
reg_model_hist = reg_model.fit(train_images, train_labels, epochs=10,
              validation_data=(test_images, test_labels))
model_list.append(reg_model_hist)
model_name_list.append("reg_model")

norm_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(10, 10)),
    tf.keras.layers.Dense(15, activation='relu'),
    tf.keras.layers.Dense(10)
])

norm_model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
norm_model_hist = norm_model.fit(train_images, train_labels, epochs=10,
              validation_data=(test_images, test_labels))
model_list.append(norm_model_hist)
model_name_list.append("norm_model")

#drawPlot_acc_loss(model_list, model_name_list,project_paths["plots"])
drawPlot_acc_loss(model_list, model_name_list, '/home/sap/IdeaProjects/XAI/April/plots')


