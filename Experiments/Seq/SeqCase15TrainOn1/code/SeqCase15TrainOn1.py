'''
Seq	"Dense                                          def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(2,activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'],)
    return model"	
epoch - 21	
skip - 3
at -7
predict using all

'''

import tensorflow as tf
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

epochs = 51

skip_steps = [
    [7,3]
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

callback_weights_pred = StoreWeights(project_paths["weights"], reg_train_steps=7,dtw_clusters=0, file_prefix ="Rweights" ,skip_array=skip_steps, weight_pred_ind=True,weighs_dtw_cluster_ind=True, replicate_csvs_at = 0)






fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0


# trimming inputs for fast processing
# train_images = train_images[0:500,5:15,5:15]
# train_labels = train_labels[0:500]
# test_images = test_images[0:50,5:15,5:15]
# test_labels = test_labels[0:50]



#plt.show()

logs = project_paths["weights"] + "/reg_model_history_log.csv"
csv_logger = CSVLogger(logs, append=True)

shutil.copy2(os.path.realpath(__file__), project_paths['code'])  # copying code file to logs

checkpoint_path = project_paths["checkpoints"] + "/weights_epoch-{epoch}.ckpt" 
restore_path = project_paths["checkpoints"] + "/weights_epoch-" + str(FirstSkip) + ".ckpt" 



callback_save_model_reg= ModelCheckpoint(filepath=checkpoint_path, save_weights_only=False,
                                                 verbose=1, period = FirstSkip)

#                             weight_pred_ind=1,
#                             skip_info=skip_info

opt = tf.keras.optimizers.Adam()

def create_model():
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    model.add(layers.Dense(2, activation='relu'))
    model.add(layers.Dense(10))
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'],
              optimizer = 'adam')
    return model


reg_model = create_model()
save_graph_plot(reg_model, project_paths["plots"] + "/reg_model.ps")
save_graph_json(reg_model, project_paths["plots"] + "/reg_model.json")


reg_model_hist = reg_model.fit(train_images, train_labels, epochs=epochs,
          validation_data=(test_images, test_labels),
          callbacks=[csv_logger,callback_save_model_reg,callback_weights_reg])

model_list.append(reg_model_hist)
model_name_list.append("Regular model ")

pred_model = tf.keras.models.load_model(restore_path)

pred_model_hist = pred_model.fit(train_images, train_labels, epochs=epochs - TotalSkips-1 ,
          initial_epoch = FirstSkip ,
          validation_data=(test_images, test_labels),
          callbacks=[csv_logger,callback_weights_pred])

model_list.append(pred_model_hist)
model_name_list.append("Pred Model ")

drawPlot_acc_loss(model_list, model_name_list,project_paths["plots"])