#Tepcoh 4 is the point where overfit started , lets see if we can use reg to avoid overfit
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback,CSVLogger
import sys
import os
import datetime
import matplotlib.pyplot as plt_loss


sys.path.append(os.getcwd())
from tools.project_tools import get_project_name, get_project_paths
from tools.model_info import save_graph_plot, save_graph_json

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
epochs = 101
batch_size = 50
validation_split = 0.2
skip_epoch = 5
skip_from = 50

project_paths = get_project_paths(sys.argv[0], to_tmp=False)
logs = project_paths["weights"] + "/reg_model_history_log.csv"
csv_logger = CSVLogger(logs, append=True)
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()


train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0




num_classes = len(np.unique(train_labels))
num_train_samples = train_images.shape[0]

def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    return model


model1 = create_model()
model1.summary()
project_paths = get_project_paths(sys.argv[0], to_tmp=False)

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
print(f'Orig --> Test loss: {test_loss} / Test accuracy: {test_acc}')


print ("<-------------------Reinitializing the model and predicting weights----------------->")
logs2 = project_paths["weights"] + "/orig_model_history_log.csv"
#command = "cp %s  %s" %(logs,logs2)
#os.system(command)
csv_logger2 = CSVLogger(logs2, append=True)
load_model = project_paths["checkpoints"] + "/weights_epoch-"+ str(skip_from) + ".ckpt"
# loading model instead of only weights 

model2 =  tf.keras.models.load_model(load_model)
save_graph_plot(model2, project_paths["plots"] + "/oriig_model.ps")
save_graph_json(model2, project_paths["weights"] + "/orig_model.json")
log_dir = project_paths["tb"] + "/orig/" + date_time
tensorboard_callback2 = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
history2 = model2.fit(train_images, train_labels, 
                    steps_per_epoch=num_train_samples / batch_size  ,  # 5 epochs per full dataset rotation
                    epochs=epochs ,
                    initial_epoch = skip_from,
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
print(f'Res --> Test loss: {test_loss} / Test accuracy: {test_acc}')



# preparing acc and loss charts 
epochs_reg = range(0,(epochs - skip_epoch))
import pdb ;pdb.set_trace()

plt_loss.plot(epochs_reg, history1.history['loss'], color='red', label='Loss Regression',linestyle='--', linewidth=3)
plt_loss.plot(epochs_reg, history1.history['val_loss'], color='green', label='Val loss Regression',linestyle='--', linewidth=3)

temp_acc = np.concatenate([history1.history['loss'][0:(skip_from-1)],history2.history['loss']])
epochs_orig = range(0,temp_acc.shape[0])
plt_loss.plot(epochs_orig, temp_acc, color='red', label='Loss Orig')
temp_acc = np.concatenate([history1.history['val_loss'][0:(skip_from-1)],history2.history['val_loss']])
plt_loss.plot(epochs_orig, temp_acc, color='green', label='Val loss Orig')
plt_loss.title('Loss')
plt_loss.xlabel('Epochs')
plt_loss.ylabel('Loss')
plt_loss.legend()
plt_loss.savefig(project_paths["weights"] + "/Loss.png")



plt_loss.clf()
plt_loss.close()

plt_loss.plot(epochs_reg, history1.history['accuracy'], color='red', label='Acc Regression',linestyle='--', linewidth=3)
plt_loss.plot(epochs_reg, history1.history['val_accuracy'], color='green', label='Val Acc Regression',linestyle='--', linewidth=3)



temp_acc = np.concatenate([history1.history['accuracy'][0:(skip_from-1)],history2.history['accuracy']])
#epochs_orig = range(0,temp_acc.shape[0])
plt_loss.plot(epochs_orig, temp_acc, color='red', label='Acc Orig')
temp_acc = np.concatenate([history1.history['val_accuracy'][0:(skip_from-1)],history2.history['val_accuracy']])
plt_loss.plot(epochs_orig,temp_acc, color='green', label='Val Acc Orig')
plt_loss.title('Accuracy')
plt_loss.xlabel('Epochs')
plt_loss.ylabel('Accuracy')
plt_loss.legend()
plt_loss.savefig(project_paths["weights"] + "/Acc.png")


