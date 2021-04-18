# purpose of this  tets os juisy tp rinm t up wyj cifar 100 and see the accuracy it can producde with gioven gpu
# enext  we will predict thss weights ovr the keras cfar linera regressio odek

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import datetime
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback,CSVLogger
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
epochs=5
batch_size=20

logs_path = "C:/Users/Sahil/IdeaProjects/thesis/code/desktop/logs/"
logs = logs_path + "model_history_log1.csv"
csv_logger = CSVLogger(logs, append=True)      
(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()
c1 = np.concatenate((train_images,test_images), axis=0)
c2 = np.concatenate((train_labels,test_labels), axis=0)
train_images = c1[:-2000]
train_labels = c2[:-2000]
test_images =  c1[-2000:]
test_labels =  c2[-2000:]



# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

num_classes = len(np.unique(train_labels))
num_train_samples = train_images.shape[0]


model = models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.summary()
model.add(layers.Flatten())
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes))

model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
log_dir = logs_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(train_images, train_labels, 
                    steps_per_epoch=num_train_samples / batch_size/20 ,  # 5 epochs per full dataset rotation
                    epochs=epochs, 
                    batch_size=batch_size,
                    validation_data=(test_images, test_labels),
					callbacks=[
						#ModelCheckpoint(
							# "C:/GPU/logs/weights_epoch-epoch02d.hdf5",
                            # save_weights_only=True,
                            # period = 1),
						csv_logger,
                        tensorboard_callback]
					)
                    


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
