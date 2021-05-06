#purpose - to skip few epochs , run predictions on weights , feed weights back,  run further 
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback,CSVLogger
import sys
import os
sys.path.append(os.getcwd())
from tools.project_tools import get_project_name, get_project_paths
from tools.model_info import save_graph_plot, save_graph_json

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
epochs=150
batch_size=50
validation_split = 0.2
skip_epoch=0
skip_from=200


#import pdb;pdb.set_trace()
project_paths = get_project_paths(sys.argv[0], to_tmp=False)
logs = project_paths["weights"] + "/model_history_log.csv"
#logs_path = "/home/pasricha/keras_log-weights/log/cifar100/weights/"
#logs = logs_path + "model_history_log1.csv"
csv_logger = CSVLogger(logs, append=True)
(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()

train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

num_classes = len(np.unique(train_labels))
num_train_samples = train_images.shape[0]

model = models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(num_classes))
model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

project_paths = get_project_paths(sys.argv[0], to_tmp=False)
save_graph_plot(model, project_paths["plots"] + "/model.ps")
save_graph_json(model, project_paths["weights"] + "/model.json")

history = model.fit(train_images, train_labels, 
                    steps_per_epoch=num_train_samples / batch_size  ,  # 5 epochs per full dataset rotation
                    epochs=epochs, 
                    batch_size=batch_size,
                    #validation_split=validation_split,
                    #validation_data=(test_images, test_labels),
					callbacks=[
						ModelCheckpoint(
							project_paths["checkpoints"] + "/weights_epoch-{epoch:02d}.hdf5",
                            save_weights_only=True,
                            period = 1,
                            skip_epoch=skip_epoch,
                        skip_from=skip_from),
						csv_logger]
					)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(f'Test loss: {test_loss} / Test accuracy: {test_acc}')
