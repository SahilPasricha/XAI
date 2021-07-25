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
import shutil
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

sys.path.append(os.getcwd())
from tools.project_tools import get_project_name, get_project_paths
from tools.compreRegenImages import writeRegenImages

from CustomCallback.storeWeightsNew import StoreWeights
from tools.create_charts import drawPlot_acc_loss
sys.path.append(os.getcwd())
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
epochs = 101
batch_size = 50
validation_split = 0.2
skip_steps = [
    [4,1],
    [9,1],
    [15,1],
    [21,1],
    [27,1],
    [33,1],
    [41,1],
    [50,1],
    [62,1],
    [74,1],
    [86,1]
]
if sum(j for i,j in skip_steps) + max(skip_steps)[0] > epochs:
    print("Exiting:skip_steps + last skip exceeds total Epochs")
    quit()

FirstSkip = skip_steps[0][0]

TotalSkips = sum([j for i,j in skip_steps])

reg_train_steps = 5
clusters = 5
latent_dim = 7
model_list = []
model_name_list = []

project_paths = get_project_paths(sys.argv[0], to_tmp=False)

callback_weights_reg = StoreWeights(project_paths["weights"], reg_train_steps=0,dtw_clusters=0, file_prefix ="Oweights" , weight_pred_ind=False,weighs_dtw_cluster_ind=False, replicate_csvs_at = FirstSkip )

callback_weights_pred = StoreWeights(project_paths["weights"], reg_train_steps=reg_train_steps,dtw_clusters=0, file_prefix ="Rweights" ,skip_array=skip_steps, weight_pred_ind=True,weighs_dtw_cluster_ind=True, replicate_csvs_at = 0)


checkpoint_path = project_paths["checkpoints"] + "/weights_epoch-{epoch}.ckpt"
restore_path = project_paths["checkpoints"] + "/weights_epoch-" + str(FirstSkip) + ".ckpt"
callback_save_model_reg= ModelCheckpoint(filepath=checkpoint_path, save_weights_only=False,
                                         verbose=1, period = FirstSkip)


project_paths = get_project_paths(sys.argv[0], to_tmp=False)
logs = project_paths["weights"] + "/orig_model_history_log.csv"
csv_logger = CSVLogger(logs, append=True)

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

shutil.copy2(os.path.realpath(__file__), project_paths['code'])  # copying code file to logs

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

model1 = create_mTotalSkipsodel()



history1 = model1.fit(x_train, x_train,
                      epochs=epochs,
                      shuffle=True,
                      validation_data=(x_test, x_test)
                      ,callbacks=[csv_logger,callback_save_model_reg,callback_weights_reg,callback_save_model_reg])
model1.summary()

model_list.append(history1)
model_name_list.append("Regular model")

print ("<-------------------Reinitializing the Orig model ----------------->")
logs2 = project_paths["weights"] + "/orig_model_history_log.csv"
csv_logger2 = CSVLogger(logs2, append=True)
load_model = restore_path
print("Loading model ", load_model)
model2 =  tf.keras.models.load_model(load_model)
#save_graph_plot(model2, project_paths["plots"] + "/orig_model.ps")
#save_graph_json(model2, project_paths["weights"] + "/orig_model.json")
pred_model_hist = model2.fit(x_train, x_train,
                      steps_per_epoch=x_train.shape[0] / batch_size  ,  # 5 epochs per full dataset rotation
                      epochs=epochs - 1 - TotalSkips ,
                      initial_epoch = FirstSkip,
                      batch_size=batch_size,
                      #validation_split=validation_split,
                      validation_data=(x_test, x_test),
                      callbacks=[csv_logger2, callback_weights_reg]
                      )



model_list.append(pred_model_hist)
model_name_list.append("Estimated Weight Model")

drawPlot_acc_loss(model_list, model_name_list,project_paths["plots"])

regen_images_orig = model1(x_test)
regen_images_pred = model2(x_test)
count = 10
writeRegenImages(x_test,regen_images_orig,regen_images_pred,count,project_paths["plots"])