#purpose - to skip few epochs , run predictions on weights , feed weights back,  run further 
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
epochs=7
batch_size=50
validation_split = 0.2
skip_epoch=0
skip_from=10
epochs= epochs-skip_epoch+1

logs_path = "/home/pasricha/keras_log-weights/log/cifar100/weights/"
logs = logs_path + "model_history_log1.csv"
csv_logger = CSVLogger(logs, append=True)
(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()

train_images=train_images[:1000]
train_labels=train_labels[:1000]
test_images=test_images[:100]
test_labels=test_labels[:100]


#Just to make it small test , Talor 

train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

num_classes = len(np.unique(train_labels))
num_train_samples = train_images.shape[0]


model = models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(num_classes))

model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(train_images, train_labels, 
                    #steps_per_epoch=num_train_samples / batch_size ,  # 5 epochs per full dataset rotation
                    epochs=epochs, 
                    batch_size=batch_size,
                    #validation_split=validation_split,
                    #validation_data=(test_images, test_labels),
					callbacks=[
						ModelCheckpoint(
							"/home/pasricha/keras_log-weights/log/cifar100/checkpoints/",
                            save_weights_only=True,
                            period = 1,
                            skip_epoch=skip_epoch,
                        skip_from=skip_from),
						csv_logger,
                        tensorboard_callback]
					)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(f'Test loss: {test_loss} / Test accuracy: {test_acc}')