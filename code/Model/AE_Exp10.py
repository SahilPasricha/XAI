#reanmed for caei_c10_ms1at7for21.py
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
import shutil
os.environ['CUDA_VISIBLE_DEVICES'] = "2"



sys.path.append(os.getcwd())
from tools.project_tools import get_project_name, get_project_paths
from CustomCallback.storeWeightsNew import StoreWeights
from tools.create_charts import drawPlot_acc_loss
from tools.compreRegenImages import writeRegenImages

from tools.project_tools import get_project_name, get_project_paths
from tools.model_info import save_graph_plot, save_graph_json
from tensorflow.keras.models import Model

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
epochs = 21
batch_size = 50
validation_split = 0.2
skip_info = [
    [4,1],
    [7,1],
    [11,1]
]
reg_train_steps = -1
# A 2d list with each array index  says #skipat , #skipfor
skip_info = sorted(skip_info,key=lambda l:l[0]) # sorted the list by first col i.e skip at
skip_epoch = sum(np.array(skip_info)[:,1])
skip_from = skip_info[0][0]
latent_dim = 7
model_list = []
model_name_list = []

project_paths = get_project_paths(sys.argv[0], to_tmp=False)
logs = project_paths["weights"] + "/reg_model_history_log.csv"
csv_logger = CSVLogger(logs, append=True)

project_paths = get_project_paths(sys.argv[0], to_tmp=False)

callback_weights_reg = StoreWeights(project_paths["weights"], reg_train_steps=0,dtw_clusters=0, file_prefix ="Oweights" , weight_pred_ind=False,weighs_dtw_cluster_ind=False, replicate_csvs_at = skip_from )

callback_weights_pred = StoreWeights(project_paths["weights"], reg_train_steps=reg_train_steps,dtw_clusters=0, file_prefix ="Rweights" ,skip_array=skip_info, weight_pred_ind=True,weighs_dtw_cluster_ind=True, replicate_csvs_at = 0)




(x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.



x_train = x_train.reshape(-1,32, 32, 1)
x_test = x_test.reshape(-1,32, 32   , 1)


csv_logger = CSVLogger(logs, append=True)

def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(1, (3, 3), activation='relu', padding='same'))

    model.compile(optimizer='adam',
                  loss=losses.MeanSquaredError(),
                  metrics=['accuracy'])
    return model

model1 = create_model()

#save_graph_plot(model1, project_paths["plots"] + "/reg_model.ps")
#save_graph_json(model1, project_paths["weights"] + "/reg_model.json")

checkpoint_path = project_paths["checkpoints"] + "/weights_epoch-{epoch}.ckpt"

project_paths = get_project_paths(sys.argv[0], to_tmp=False)


history1 = model1.fit(x_train, x_train,
                      epochs=epochs-skip_epoch,
                      shuffle=True,
                      validation_data=(x_test, x_test)
                      ,callbacks=[
        csv_logger,
        ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=False,
            period = 1,
            weight_pred_ind=1,
            skip_info=skip_info),
        callback_weights_pred])

model1.summary()
model_list.append(history1)
model_name_list.append("Estimated Weight Model")

print ("<-------------------Reinitializing the Orig model ----------------->")
logs2 = project_paths["weights"] + "/orig_model_history_log.csv"
csv_logger2 = CSVLogger(logs2, append=True)
load_model = project_paths["checkpoints"] + "/weights_epoch-"+ str(skip_from) + ".ckpt" # -1 is adjusted to pick the checkpoint just before reg , the epoch counter starts from 0 is adjusted with ckpt stored from 1
print("Loading model ")
print(load_model)
model2 =  tf.keras.models.load_model(load_model)
#save_graph_plot(model2, project_paths["plots"] + "/orig_model.ps")
#save_graph_json(model2, project_paths["weights"] + "/orig_model.json")
history2 = model2.fit(x_train, x_train,
                      steps_per_epoch=x_train.shape[0] / batch_size  ,  # 5 epochs per full dataset rotation
                      epochs=epochs ,
                      initial_epoch = skip_from,
                      batch_size=batch_size,
                      #validation_split=validation_split,
                      validation_data=(x_test, x_test),
                      callbacks=[
                          ModelCheckpoint(
                              project_paths["checkpoints"] + "/orig_weights_epoch-{epoch:02d}.hdf5",
                              save_weights_only=False,
                              period = 1)
                          ,csv_logger2,
                          callback_weights_reg
                          #,tensorboard_callback2
                      ]
                      )
model2.summary()
model_list.append(history2)
model_name_list.append("Regular model")


drawPlot_acc_loss(model_list, model_name_list,project_paths["plots"])

regen_images_orig = model2(x_test)
regen_images_pred = model1(x_test)
count = 10
writeRegenImages(x_test,regen_images_orig,regen_images_pred,count,project_paths["plots"])