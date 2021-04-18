# Build MNIST to late est Pred on it . thjis being simpler main revela clear impacts of pred on it 

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback,CSVLogger
import sys
import os
import datetime
import numpy as np
sys.path.append(os.getcwd())
from tools.project_tools import get_project_name, get_project_paths
from tools.model_info import save_graph_plot, save_graph_json


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
epochs = 21
batch_size = 50
validation_split = 0.2
skip_epoch = 3
skip_from = 13

tfds.disable_progress_bar()
tf.enable_v2_behavior()
project_paths = get_project_paths(sys.argv[0], to_tmp=False)
logs = project_paths["weights"] + "/reg_model_history_log.csv"
csv_logger = CSVLogger(logs, append=True)
(train_images, train_labels), (test_images, test_labels) =  tf.keras.datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
num_classes = len(np.unique(train_labels))
num_train_samples = train_images.shape[0]

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(2,activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'],)
    return model

model1 = create_model()
model1.summary()
save_graph_plot(model1, project_paths["plots"] + "/reg_model.ps")
save_graph_json(model1, project_paths["weights"] + "/reg_model.json")
date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = project_paths["tb"] + "/reg/" + date_time
tensorboard_callback1 = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
checkpoint_path = project_paths["checkpoints"] + "/weights_epoch-{epoch}.ckpt" 
checkpoint_dir = os.path.dirname(checkpoint_path)

history1 = model1.fit(train_images, train_labels, 
                    steps_per_epoch=num_train_samples / batch_size  ,  # 5 epochs per full dataset rotation
                    epochs = (epochs - skip_epoch), 
                    batch_size=batch_size,
                    #validation_split=validation_split,
                    validation_data=(test_images, test_labels),
					callbacks=[
						ModelCheckpoint(
							filepath=checkpoint_path, 
                            save_weights_only=False,
                            period = 1,
                            weight_pred_ind=1,
                            skip_from = skip_from,
                            skip_epoch = skip_epoch),
						csv_logger,
                        tensorboard_callback1
                        ]
					)
test_loss, test_acc = model1.evaluate(test_images,  test_labels, verbose=2)
print(f'Ref --> Test loss: {test_loss} / Test accuracy: {test_acc}')



print ("<-------------------Reinitializing the Orig model ----------------->")
logs2 = project_paths["weights"] + "/orig_model_history_log.csv"
csv_logger2 = CSVLogger(logs2, append=True)
load_model = project_paths["checkpoints"] + "/weights_epoch-"+ str(skip_from) + ".ckpt"
print("Loading model")
model2 =  tf.keras.models.load_model(load_model)
save_graph_plot(model2, project_paths["plots"] + "/orig_model.ps")
save_graph_json(model2, project_paths["weights"] + "/orig_model.json")
log_dir2 = project_paths["tb"] + "/orig/" + date_time
tensorboard_callback2 = tf.keras.callbacks.TensorBoard(log_dir=log_dir2, histogram_freq=1)
history2 = model2.fit(train_images, train_labels, 
                    steps_per_epoch=num_train_samples / batch_size  ,  # 5 epochs per full dataset rotation
                    epochs=epochs-1 ,
                    initial_epoch = skip_from-1,
                    batch_size=batch_size,
                    #validation_split=validation_split,
                    validation_data=(test_images, test_labels),
					callbacks=[
						ModelCheckpoint(
							project_paths["checkpoints"] + "/orig_weights_epoch-{epoch:02d}.hdf5",
                            save_weights_only=False,
                            period = 1),
						csv_logger2,
                        tensorboard_callback2
                        ]
					)
test_loss, test_acc = model2.evaluate(test_images,  test_labels, verbose=2)
print(f'Orig --> Test loss: {test_loss} / Test accuracy: {test_acc}')



