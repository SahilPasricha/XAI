#Tepcoh 4 is the point where overfit started , lets see if we can use reg to avoid overfit
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

sys.path.append(os.getcwd())
from tools.project_tools import get_project_name, get_project_paths


from tools.project_tools import get_project_name, get_project_paths
from tools.model_info import save_graph_plot, save_graph_json
from tensorflow.keras.models import Model

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
epochs = 51
batch_size = 50
validation_split = 0.2
skip_info = [
             [4,1],
             [7,1], 
            ]
# A 2d list with each array index  says #skipat , #skipfor 
skip_info = sorted(skip_info,key=lambda l:l[0]) # sorted the list by first col i.e skip at 
skip_epoch = sum(np.array(skip_info)[:,1])
skip_from = skip_info[0][0]
latent_dim = 7

project_paths = get_project_paths(sys.argv[0], to_tmp=False)
logs = project_paths["weights"] + "/reg_model_history_log.csv"
csv_logger = CSVLogger(logs, append=True)



(x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()


x_train = x_train/255
x_test = x_test/255

# x_train = x_train.reshape(-1,28, 28, 1)
# x_test = x_test.reshape(-1,28, 28, 1)

csv_logger = CSVLogger(logs, append=True)

def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization())     # 32x32x32
    model.add(layers.Conv2D(16, kernel_size=3, strides=2, padding='same', activation='relu'))      # 16x16x32
    model.add(layers.Conv2D(16, kernel_size=3, strides=1, padding='same', activation='relu'))      # 16x16x32
    model.add(layers.BatchNormalization())     # 16x16x32
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu'))      # 32x32x32
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(3,  kernel_size=1, strides=1, padding='same', activation='sigmoid'))   # 32x32x3

    model.compile(optimizer='adam', metrics=['accuracy'], loss='mean_squared_error')
    model.summary()
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
                            skip_info=skip_info)])

model1.summary()
predicted_images_reg = model1(x_test)
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
						,csv_logger2
                        #,tensorboard_callback2
                        ]
					)
model2.summary()
predicted_images_orig = model2(x_test)
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
n = 14
import random
plt.figure(figsize=(30, 5))
for i in range(n):
    random_image = random.randint(0, x_test.shape[0]-1)
    # display original
    ax = plt.subplot(3,n,i+1)
    plt.imshow(x_test[random_image])
    plt.title("original_"+str(random_image))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction original
    ax = plt.subplot(3,n,n+i+1)
    plt.imshow(predicted_images_orig[random_image])
    plt.title("reconstructed_"+str(random_image))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display reconstruction after regression 
    ax = plt.subplot(3, n, n*2 + i + 1)
    plt.imshow(predicted_images_reg[random_image])
    plt.title("reg. rec...ed_"+str(random_image))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
plt.savefig(project_paths["weights"] + "/ae-images.png")
