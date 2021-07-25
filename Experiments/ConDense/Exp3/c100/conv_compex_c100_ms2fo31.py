#Tepcoh 4 is the point where overfit started , lets see if we can use reg to avoid overfit
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D,Activation,Dropout
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

from tensorflow.keras import utils
# Convert correct answer labels of y_train,y_test to One-Hot

from tools.project_tools import get_project_name, get_project_paths
from tools.model_info import save_graph_plot, save_graph_json
from tensorflow.keras.models import Model
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam

optimizer = Adam()
img_width, img_height, img_num_channels = 32, 32, 3
loss_function = sparse_categorical_crossentropy
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
epochs = 31
batch_size = 50
validation_split = 0.2
skip_info = [
             [5,1],
             [11,1]
            ]
# A 2d list with each array index  says #skipat , #skipfor 
skip_info = sorted(skip_info,key=lambda l:l[0]) # sorted the list by first col i.e skip at 
skip_epoch = sum(np.array(skip_info)[:,1])
skip_from = skip_info[0][0]
latent_dim = 7

project_paths = get_project_paths(sys.argv[0], to_tmp=False)
logs = project_paths["weights"] + "/reg_model_history_log.csv"
csv_logger = CSVLogger(logs, append=True)

# Load CIFAR-100 data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data()


# Determine shape of the data
input_shape = (img_width, img_height, img_num_channels)

no_classes = 100

train_images = train_images/255
test_images = test_images/255

train_labels = utils.to_categorical(train_labels,100)
test_labels = utils.to_categorical(test_labels,100)


csv_logger = CSVLogger(logs, append=True)

def create_model():
    model = models.Sequential()
    model.add(Conv2D(32,(3,3),padding='same',input_shape=(32,32,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100,activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model


model1 = create_model()

#save_graph_plot(model1, project_paths["plots"] + "/reg_model.ps")
#save_graph_json(model1, project_paths["weights"] + "/reg_model.json")

checkpoint_path = project_paths["checkpoints"] + "/weights_epoch-{epoch}.ckpt" 

project_paths = get_project_paths(sys.argv[0], to_tmp=False)


history1 = model1.fit(train_images, train_labels,
                epochs=epochs-skip_epoch,
                shuffle=True,
                batch_size=128,      
                validation_data=(test_images,test_labels)
               ,callbacks=[
                        csv_logger,
						ModelCheckpoint(
							filepath=checkpoint_path,  
                            save_weights_only=False,
                            period = 1,
                            weight_pred_ind=1,
                            skip_info=skip_info)])

model1.summary()
#predicted_images_reg = model1(test_images,test_labels)
import pdb;pdb.set_trace()
print ("<-------------------Reinitializing the Orig model ----------------->")
logs2 = project_paths["weights"] + "/orig_model_history_log.csv"
csv_logger2 = CSVLogger(logs2, append=True)
load_model = project_paths["checkpoints"] + "/weights_epoch-"+ str(skip_from) + ".ckpt" # -1 is adjusted to pick the checkpoint just before reg , the epoch counter starts from 0 is adjusted with ckpt stored from 1
print("Loading model ")
print(load_model)
model2 =  tf.keras.models.load_model(load_model)
#save_graph_plot(model2, project_paths["plots"] + "/orig_model.ps")
#save_graph_json(model2, project_paths["weights"] + "/orig_model.json")
history2 = model2.fit(train_images, train_labels, 
                    epochs=epochs ,
                    initial_epoch = skip_from,
                    batch_size=128,      
                    #validation_split=validation_split,
                    validation_data=(test_images,test_labels),
					callbacks=[
						ModelCheckpoint(
							project_paths["checkpoints"] + "/orig_weights_epoch-{epoch:02d}.hdf5",
                            save_weights_only=False,
                            period = 1)
						,csv_logger2
                        #,tensorboard_callback2
                        ]
					)
model2.summary()
#predicted_images_orig = model2(test_images,test_labels)
epochs_reg = range(0,len(history1.history['loss']))
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
plt_loss.savefig(project_paths["weights"] + "/Loss.png")



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
plt_loss.savefig(project_paths["weights"] + "/Acc.png")

plt.clf()
plt.close()
# draw images plot to compare , orid, predicted and reg predicted images
#commenetd for non AE model 
#n = 14
# import random
# plt.figure(figsize=(30, 5))
# for i in range(n):
#     random_image = random.randint(0, x_test.shape[0]-1)
#     # display original
#     ax = plt.subplot(3,n,i+1)
#     plt.imshow(x_test[random_image])
#     plt.title("original_"+str(random_image))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

#     # display reconstruction original
#     ax = plt.subplot(3,n,n+i+1)
#     plt.imshow(predicted_images_orig[random_image])
#     plt.title("reconstructed_"+str(random_image))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
    
#     # display reconstruction after regression 
#     ax = plt.subplot(3, n, n*2 + i + 1)
#     plt.imshow(predicted_images_reg[random_image])
#     plt.title("reg. rec...ed_"+str(random_image))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()
# plt.savefig(project_paths["weights"] + "/ae-images.png")
