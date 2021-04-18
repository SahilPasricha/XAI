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
from tools.project_tools import get_project_name, get_project_paths
from tools.model_info import save_graph_plot, save_graph_json
from CustomCallback.StoringWeights import StoreWeights

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import shutil




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
reg_train_steps = 3
clusters = 200
project_paths = get_project_paths(sys.argv[0], to_tmp=False)
callback_weights = StoreWeights(project_paths["weights"], reg_train_steps=reg_train_steps, dtw_clusters=clusters,
                                skip_array=skip_steps, weight_pred_ind=True, weighs_dtw_cluster_ind=True)


print(tf.__version__)
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

# trimming inputs for fast processing
train_images = train_images[0:5000]
train_labels = train_labels[0:5000]
test_images = test_images[0:500]
test_labels = test_labels[0:500]

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
#plt.show()

logs = project_paths["weights"] + "/reg_model_history_log.csv"
shutil.copy2(os.path.realpath(__file__), project_paths['code'])  # copying code file to logs

checkpoint_path = "training_1/cp.ckpt"  # redefine
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Weights mechanism
# skip_steps [skip at epoch number, skip these number of steps]

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback, callback_weights])
