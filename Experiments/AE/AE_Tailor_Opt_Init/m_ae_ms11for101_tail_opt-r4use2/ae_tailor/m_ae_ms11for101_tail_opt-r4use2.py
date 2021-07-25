#Purpose -- Tailor optimizer and initializer(Later)
# Reg Param - Skip 4 , reg using all 
# To be compared to m_ae_ms11for101_images-reg4-useall.py
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
epochs = 101
batch_size = 50
validation_split = 0.2
skip_info = [
             [4,4],
             [9,4],
             [15,4],
             [21,4],
             [27,4], 
             [33,4], 
             [41,4],
             [50,4],
             [62,4]
            ]
# A 2d list with each array index  says #skipat , #skipfor 
skip_info = sorted(skip_info,key=lambda l:l[0]) # sorted the list by first col i.e skip at 
skip_epoch = sum(np.array(skip_info)[:,1])
skip_from = skip_info[0][0]
latent_dim = 7

if(skip_info[len(skip_info)-1][0] + skip_epoch > epochs):
    print("More skips then model's life cycle")
    sys.exit()
    
  

project_paths = get_project_paths(sys.argv[0], to_tmp=False)
logs = project_paths["weights"] + "/reg_model_history_log.csv"
csv_logger = CSVLogger(logs, append=True)
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()


train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.




num_classes = len(np.unique(train_labels))
num_train_samples = train_images.shape[0]


csv_logger = CSVLogger(logs, append=True)

opt = tf.keras.optimizers.Adam(learning_rate=0.01)

def create_model():
    model = models.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(latent_dim, activation='relu'))
    model.add(layers.Dense(784, activation='sigmoid'))
    model.add(layers.Reshape((28,28)))
    model.compile(loss=losses.MeanSquaredError(),
              metrics=['accuracy'],
              optimizer = opt)
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
n = 10
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

