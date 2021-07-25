'''
Model1--> Normal
Model 2 --> DTW , multi skip , predict - 1, Train 3
'''



import tensorflow as tf
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

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import shutil

model_list = []
model_name_list = []
epochs = 101
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images[0:500,5:15,5:15]
train_labels = train_labels[0:500]
test_images = test_images[0:50,5:15,5:15]
test_labels = test_labels[0:50]

checkpoint_path = "training_1/cp.ckpt"  # redefine
checkpoint_dir = os.path.dirname(checkpoint_path)



# Weights mechanism
# skip_steps [skip at epoch number, skip these number of steps]

norm_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(10, 10)),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(10)
])


project_paths = get_project_paths(sys.argv[0], to_tmp=False)



skip_steps = [[27, 1], [33, 1], [41, 1], [50, 1], [62, 1], [74, 1], [86, 1] ]
#skip_steps =  [ [4, 1]]


TotalSkips = sum([j for i,j in skip_steps])
if sum(j for i,j in skip_steps) + max(skip_steps)[0] > epochs:
    print("Exiting:skip_steps + last skip exceeds total Epochs")
    quit()


FirstSkip = skip_steps[0][0]
#Callbacks
reg_train_steps = 3
clusters = 20

callback_weights_reg = StoreWeights(project_paths["weights"], reg_train_steps=0,dtw_clusters=0, file_prefix ="Oweights" ,skip_array=skip_steps, weight_pred_ind=False,weighs_dtw_cluster_ind=False, replicate_csvs_at = FirstSkip )

callback_weights_pred = StoreWeights(project_paths["weights"], reg_train_steps=reg_train_steps,dtw_clusters=0, file_prefix ="Rweights" ,skip_array=skip_steps, weight_pred_ind=True,weighs_dtw_cluster_ind=False, replicate_csvs_at = 0)

callback_weights_cluster_pred = StoreWeights(project_paths["weights"], reg_train_steps=reg_train_steps, dtw_clusters=clusters, file_prefix ="Cweights" ,skip_array=skip_steps, weight_pred_ind=True, weighs_dtw_cluster_ind=True, replicate_csvs_at = 0)

checkpoint_path = project_paths["checkpoints"] + "/weights_epoch-{epoch}.ckpt"
restore_path = project_paths["checkpoints"] + "/weights_epoch-" + str(FirstSkip) + ".ckpt"
callback_save_model_reg= ModelCheckpoint(filepath=checkpoint_path, save_weights_only=False,
                                         verbose=1, period = FirstSkip)


logs_norm = project_paths["weights"] + "/" + "RegModelE" + str(epochs) + "Skp.csv"
callback_weights_no_pred = StoreWeights(project_paths["weights"], reg_train_steps=0, dtw_clusters=0, file_prefix ="Oweights" ,skip_array=[], weight_pred_ind=False, weighs_dtw_cluster_ind=True)

norm_model.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
norm_model_hist = norm_model.fit(train_images, train_labels, epochs=epochs,
                                 validation_data=(test_images, test_labels),
                                 callbacks=[CSVLogger(logs_norm, append=True),callback_weights_reg, callback_save_model_reg])
# save_graph_plot(norm_model, project_paths["plots"] + "/norm_model.ps")
# save_graph_json(norm_model, project_paths["plots"] + "/norm_model.json")
model_list.append(norm_model_hist)
model_name_list.append("Regular model")

print("******************Weight Estimaiton Model************************************")

logs_estimation = project_paths["weights"] + "/" + "EstModelE" + str(epochs) + "Skp" + str(TotalSkips) + ".csv"
load_model = restore_path
print("Loading model ", load_model)
reg_model =  tf.keras.models.load_model(load_model)

reg_model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

reg_model_hist = reg_model.fit(train_images, train_labels,epochs = epochs - 1 - TotalSkips ,  initial_epoch = FirstSkip,
                               validation_data=(test_images, test_labels),
                               callbacks=[CSVLogger(logs_estimation, append=True), callback_weights_pred])

# save_graph_plot(reg_model, project_paths["plots"] + "/.ps")
# save_graph_json(reg_model, project_paths["plots"] + "/reg_model.json")
model_list.append(reg_model_hist)
model_name_list.append("Individual Wt. Est. Model")

print("******************Cluster Weight Estimaiton Model************************************")

logs_clus_est = project_paths["weights"] + "/" + "EstClusterModelE" + str(epochs) + "Skp" + str(TotalSkips) + ".csv"
load_model = restore_path
print("Loading model ", load_model)
reg_clus_model =  tf.keras.models.load_model(load_model)

reg_clus_model.compile(optimizer='adam',
                       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                       metrics=['accuracy'])

reg_clus_model_hist = reg_clus_model.fit(train_images, train_labels, epochs = epochs - 1 - TotalSkips ,  initial_epoch = FirstSkip,
                                         validation_data=(test_images, test_labels),
                                         callbacks=[CSVLogger(logs_clus_est, append=True), callback_weights_cluster_pred])

# save_graph_plot(reg_clus_model, project_paths["plots"] + "/.ps")
# save_graph_json(reg_clus_model, project_paths["plots"] + "/reg_model.json")
model_list.append(reg_clus_model_hist)
model_name_list.append("Cluster Wt. Est. Model")


drawPlot_acc_loss(model_list, model_name_list, project_paths["plots"])
#drawPlot_acc_loss(model_list, model_name_list, '/home/sap/IdeaProjects/XAI/April/plots')