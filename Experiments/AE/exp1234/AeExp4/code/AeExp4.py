#show gap between est and real model acc at highest 
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback,CSVLogger
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import datetime
import matplotlib.pyplot as plt_loss
import shutil
sys.path.append(os.getcwd())
from tools.project_tools import get_project_name, get_project_paths


from tools.project_tools import get_project_name, get_project_paths
from tools.model_info import save_graph_plot, save_graph_json
from tensorflow.keras.models import Model
from CustomCallback.storeWeightsNew import StoreWeights
from tools.create_charts import drawPlot_acc_loss

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
epochs = 51
batch_size = 50
validation_split = 0.2
latent_dim = 14

model_list = []
model_name_list = []

skip_steps = [
    [10,3]
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

callback_weights_pred = StoreWeights(project_paths["weights"], reg_train_steps=reg_train_steps,dtw_clusters=0, file_prefix ="Rweights" ,skip_array=skip_steps, weight_pred_ind=True,weighs_dtw_cluster_ind=True, replicate_csvs_at = 0)


checkpoint_path = project_paths["checkpoints"] + "/weights_epoch-{epoch}.ckpt"
restore_path = project_paths["checkpoints"] + "/weights_epoch-" + str(FirstSkip) + ".ckpt"
callback_save_model_reg= ModelCheckpoint(filepath=checkpoint_path, save_weights_only=False,
                                         verbose=1, period = FirstSkip)


project_paths = get_project_paths(sys.argv[0], to_tmp=False)
logs = project_paths["weights"] + "/reg_model_history_log.csv"
csv_logger = CSVLogger(logs, append=True)



(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

csv_logger = CSVLogger(logs, append=True)
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

model1 = create_model()

#save_graph_plot(model1, project_paths["plots"] + "/reg_model.ps")
#save_graph_json(model1, project_paths["weights"] + "/reg_model.json")

checkpoint_path = project_paths["checkpoints"] + "/weights_epoch-{epoch}.ckpt"

project_paths = get_project_paths(sys.argv[0], to_tmp=False)


history1 = model1.fit(x_train, x_train,
                      epochs=epochs,
                      shuffle=True,
                      validation_data=(x_test, x_test)
                      ,callbacks=[
        csv_logger,callback_weights_reg,callback_save_model_reg]
       )

model1.summary()

model_list.append(history1)
model_name_list.append("Regular model")
print ("<-------------------Reinitializing the Orig model ----------------->")
logs2 = project_paths["weights"] + "/orig_model_history_log.csv"
csv_logger2 = CSVLogger(logs2, append=True)
print("Loading model ")
model2 = tf.keras.models.load_model(restore_path)
#save_graph_plot(model2, project_paths["plots"] + "/orig_model.ps")
#save_graph_json(model2, project_paths["weights"] + "/orig_model.json")
history2 = model2.fit(x_train, x_train,
                      steps_per_epoch=x_train.shape[0] / batch_size  ,  # 5 epochs per full dataset rotation
                      epochs=epochs - TotalSkips-1 ,
                      initial_epoch = FirstSkip ,
                      batch_size=batch_size,
                      #validation_split=validation_split,
                      validation_data=(x_test, x_test),
                      callbacks=[csv_logger, callback_weights_pred]
                      )
model2.summary()

model_list.append(history2)
model_name_list.append("Pred Model ")

drawPlot_acc_loss(model_list, model_name_list,project_paths["plots"])


epochs_reg = range(0,len(history1.history['loss']))

skip_from = 3
plt_loss.plot(epochs_reg, history1.history['loss'], color='red', label='Loss Regression',linestyle='--', linewidth=3)
plt_loss.plot(epochs_reg, history1.history['val_loss'], color='green', label='Val loss Regression',linestyle='--', linewidth=3)

temp_acc = np.concatenate([history1.history['loss'][0:(skip_from)],history2.history['loss']])
epochs_orig = range(0,temp_acc.shape[0])
plt_loss.plot(epochs_orig, temp_acc, color='red', label='Loss Orig')
temp_acc = np.concatenate([history1.history['val_loss'][0:(skip_from)],history2.history['val_loss']])
plt_loss.plot(epochs_orig, temp_acc, color='green', label='Val loss Orig')
plt_loss.title('Loss')
plt_loss.xlabel('Epochs')
plt_loss.ylabel('Loss')
plt_loss.legend()
plt_loss.savefig(project_paths["weights"] + "/Loss1.png")



plt_loss.clf()
plt_loss.close()

plt_loss.plot(epochs_reg, history1.history['accuracy'], color='red', label='Acc Regression',linestyle='--', linewidth=3)
plt_loss.plot(epochs_reg, history1.history['val_accuracy'], color='green', label='Val Acc Regression',linestyle='--', linewidth=3)



temp_acc = np.concatenate([history1.history['accuracy'][0:(skip_from)],history2.history['accuracy']])
#epochs_orig = range(0,temp_acc.shape[0])
plt_loss.plot(epochs_orig, temp_acc, color='red', label='Acc Orig')
temp_acc = np.concatenate([history1.history['val_accuracy'][0:(skip_from)],history2.history['val_accuracy']])
plt_loss.plot(epochs_orig,temp_acc, color='green', label='Val Acc Orig')
plt_loss.title('Accuracy')
plt_loss.xlabel('Epochs')
plt_loss.ylabel('Accuracy')
plt_loss.legend()
plt_loss.savefig(project_paths["weights"] + "/Acc1.png")