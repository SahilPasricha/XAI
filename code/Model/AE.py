#Tepcoh 4 is the point where overfit started , lets see if we can use reg to avoid overfit
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback,CSVLogger


import sys
import os
import datetime
import matplotlib.pyplot as plt_loss
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
sys.path.append(os.getcwd())
from tools.project_tools import get_project_name, get_project_paths


from tools.project_tools import get_project_name, get_project_paths
from tools.model_info import save_graph_plot, save_graph_json
from tensorflow.keras.models import Model
from tools.create_charts import drawPlot_acc_loss
from CustomCallback.storeWeightsNew import StoreWeights

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import shutil

model_list = []
model_name_list = []

epochs = 21

skip_steps = [
    [3,1]
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

callback_weights_pred = StoreWeights(project_paths["weights"], reg_train_steps=reg_train_steps,dtw_clusters=0, file_prefix ="Rweights" ,skip_array=skip_steps, weight_pred_ind=True,weighs_dtw_cluster_ind=True, replicate_csvs_at = 0)


checkpoint_path = project_paths["checkpoints"] + "/weights_epoch-{epoch}.ckpt"
restore_path = project_paths["checkpoints"] + "/weights_epoch-" + str(FirstSkip) + ".ckpt"
callback_save_model_reg= ModelCheckpoint(filepath=checkpoint_path, save_weights_only=False,
                                         verbose=1, period = FirstSkip)
batch_size = 50
validation_split = 0.2
latent_dim = 7

project_paths = get_project_paths(sys.argv[0], to_tmp=False)
logs = project_paths["weights"] + "/reg_model_history_log.csv"
csv_logger = CSVLogger(logs, append=True)



(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

csv_logger = CSVLogger(logs, append=True)

def create_model():
    model = models.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(latent_dim, activation='relu'))
    model.add(layers.Dense(784, activation='sigmoid'))
    model.add(layers.Reshape((28,28)))
    model.compile(optimizer='adam',
                  loss=losses.MeanSquaredError(),
                  metrics=['accuracy'])
    return model



reg_model = create_model()

project_paths = get_project_paths(sys.argv[0], to_tmp=False)
# save_graph_plot(reg_model, project_paths["plots"] + "/reg_model.ps")
# save_graph_json(reg_model, project_paths["weights"] + "/reg_model.json")


reg_model_hist = reg_model.fit(x_train, x_train,
                               steps_per_epoch=x_train.shape[0] / batch_size  ,
                               epochs=epochs,
                               shuffle=True,
                               validation_data=(x_test, x_test)
                               , callbacks=[csv_logger, callback_save_model_reg, callback_weights_reg])


model_list.append(reg_model_hist)
model_name_list.append("Regular model")


print ("<-------------------Reinitializing the Orig model ----------------->")
logs2 = project_paths["weights"] + "/orig_model_history_log.csv"
csv_logger2 = CSVLogger(logs2, append=True)
print("Loading model ",restore_path)
pred_model = tf.keras.models.load_model(restore_path)
#save_graph_plot(model2, project_paths["plots"] + "/orig_model.ps")
#save_graph_json(model2, project_paths["weights"] + "/orig_model.json")
pred_model_hist = pred_model.fit(x_train, x_train,
                                 steps_per_epoch=x_train.shape[0] / batch_size,  # 5 epochs per full dataset rotation
                                 epochs=epochs - TotalSkips-1 ,
                                 initial_epoch = FirstSkip ,
                                 batch_size=batch_size,
                                 #validation_split=validation_split,
                                 validation_data=(x_test, x_test),
                                 callbacks=[csv_logger,callback_weights_pred]
                                 )
pred_model = tf.keras.models.load_model(restore_path)

model_list.append(pred_model_hist)
model_name_list.append("Pred Model ")

drawPlot_acc_loss(model_list, model_name_list,project_paths["plots"])