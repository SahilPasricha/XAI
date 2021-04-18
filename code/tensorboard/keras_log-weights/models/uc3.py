#https://towardsdatascience.com/handling-overfitting-in-deep-learning-models-c760ee047c6e

# Basic packages
# how viz looks of 2 diffrent models one reg and other overfitting 
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from keras import layers
from keras import regularizers

print(tf.__version__)



import pandas as pd 
import numpy as np
import re
import collections
import matplotlib.pyplot as plt

import nltk
nltk.download('stopwords')
    
    
from pathlib import Path
# Packages for data preparation
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from keras.callbacks import ModelCheckpoint, LambdaCallback,CSVLogger
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
# Packages for modeling
from keras import models
from keras import layers
from keras import regularizers

import sys
import os
sys.path.append(os.getcwd())
from tools.multi_model_info import save_graph_plot, save_graph_json
from tools.project_tools import get_project_name, get_project_paths
#from tensorflow.keras.callbacks import ModelCheckpoint
NB_WORDS = 100  # Parameter indicating the number of words we'll put in the dictionary
NB_START_EPOCHS = 151  # Number of epochs we usually start to train with
BATCH_SIZE = 512  # Size of the batches used in the mini-batch gradient descent
MAX_LEN = 500  # Maximum number of words in a sequence
root = Path('../')
input_path = root / 'input/' 
ouput_path = root / 'output/'
source_path = root / 'source/'
logs_path = "/home/pasricha/keras_log-weights/log/UC3/"


def deep_model(model, X_train, y_train, X_valid, y_valid):
    logs = logs_path + "model_history_log.csv"
    csv_logger = CSVLogger(logs, append=True)
    '''
    Function to train a multi-class model. The number of epochs and 
    batch_size are set by the constants at the top of the
    notebook. 
    
    Parameters:
        model : model with the chosen architecture
        X_train : training features
        y_train : training target
        X_valid : validation features
        Y_valid : validation target
    Output:
        model training history
    '''
    model.compile(optimizer='rmsprop'
                  , loss='categorical_crossentropy'
                  , metrics=['accuracy'])
    argv = "deep.py"
    project_paths = get_project_paths(argv[0], to_tmp=False)

    save_graph_plot(model, project_paths["plots"] + "/model.ps")
    save_graph_json(model, project_paths["graphs"] + "/model.json")
    #import pdb; pdb.set_trace()
    history = model.fit(X_train
                       , y_train
                       , epochs=NB_START_EPOCHS
                       , batch_size=BATCH_SIZE
                       , validation_data=(X_valid, y_valid)
                       , callbacks=[
#                            ModelCheckpoint(
#                            project_paths["checkpoints"] + "/weights_epoch-{epoch:02d}.hdf5",
#                            save_weights_only=True,
#                            period = 1),
                                    csv_logger]
#                        , callbacks = [ModelCheckpoint(project_paths["checkpoints"] + "/weights_epoch-{epoch:02d}.hdf5",
#                                                       monitor='val_acc', verbose=1,  save_weights_only=True,  period=1)] 
                       , verbose=1)
    return history
def eval_metric(model, history, metric_name):
    '''
    Function to evaluate a trained model on a chosen metric. 
    Training and validation metric are plotted in a
    line chart for each epoch.
    
    Parameters:
        history : model training history
        metric_name : loss or accuracy
    Output:
        line chart with epochs of x-axis and metric on
        y-axis
    '''
    #import pdb;
    #pdb.set_trace()
    plt.clf()
    metric = history.history[metric_name]
    val_metric = history.history['val_' + metric_name]
    e = range(1, NB_START_EPOCHS + 1)
    plt.plot(e, metric, 'bo', label='Train ' + metric_name)
    plt.plot(e, val_metric, 'b', label='Validation ' + metric_name)
    plt.xlabel('Epoch number')
    plt.ylabel(metric_name)
    plt.title('Comparing Loss training and validation for ' + model.name)
    plt.legend()
    plt.show()
    plt.savefig(logs_path  + model.name + "_" + metric_name)
    
    

    
def test_model(model, X_train, y_train, X_test, y_test, epoch_stop):
    '''
    Function to test the model on new data after training it
    on the full training data with the optimal number of epochs.
    
    Parameters:
        model : trained model
        X_train : training features
        y_train : training target
        X_test : test features
        y_test : test target
        epochs : optimal number of epochs
    Output:
        test accuracy and test loss
    '''
    model.fit(X_train
              , y_train
              , epochs=epoch_stop
              , batch_size=BATCH_SIZE
              , verbose=0)
    results = model.evaluate(X_test, y_test)
    print()
    print('Test accuracy for ' + model.name + ' : {0:.2f}%'.format(results[1]*100))
    return results
    
def remove_stopwords(input_text):
    '''
    Function to remove English stopwords from a Pandas Series.
    
    Parameters:
        input_text : text to clean
    Output:
        cleaned Pandas Series 
    '''
    stopwords_list = stopwords.words('english')
    # Some words which might indicate a certain sentiment are kept via a whitelist
    whitelist = ["n't", "not", "no"]
    words = input_text.split() 
    clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 
    return " ".join(clean_words) 
    
def remove_mentions(input_text):
    '''
    Function to remove mentions, preceded by @, in a Pandas Series
    
    Parameters:
        input_text : text to clean
    Output:
        cleaned Pandas Series 
    '''
    return re.sub(r'@\w+', '', input_text)
def compare_models_by_metric(model_1, model_2, model_hist_1, model_hist_2, metric):
    '''
    Function to compare a metric between two models 
    
    Parameters:
        model_hist_1 : training history of model 1
        model_hist_2 : training history of model 2
        metrix : metric to compare, loss, acc, val_loss or val_acc
        
    Output:
        plot of metrics of both models
    '''
    plt.clf() # To Erase previous plots
    #import pdb;pdb.set_trace()
    metric_model_1 = model_hist_1.history[metric]
    metric_model_2 = model_hist_2.history[metric]
    e = range(1, NB_START_EPOCHS + 1)
    
    metrics_dict = {
        'acc' : 'Training Accuracy',
        'loss' : 'Training Loss',
        'val_acc' : 'Validation accuracy',
        'val_loss' : 'Validation loss',
        'val_accuracy' :  'Validation accuracy'
    }
    metric_label = metrics_dict[metric]
    plt.plot(e, metric_model_1, 'bo', label=model_1.name)
    plt.plot(e, metric_model_2, 'b', label=model_2.name)
    plt.xlabel('Epoch number')
    plt.ylabel(metric_label)
    plt.title('Comparing ' + metric_label + ' between models')
    plt.legend()
    plt.show()
    plt.savefig(logs_path  + model_1.name + model_2.name)
    plt.clf()
    
def optimal_epoch(model_hist):
    '''
    Function to return the epoch number where the validation loss is
    at its minimum
    
    Parameters:
        model_hist : training history of model
    Output:
        epoch number with minimum validation loss
    '''
    min_epoch = np.argmin(model_hist.history['val_loss']) + 1
    print("Minimum validation loss reached in epoch {}".format(min_epoch))
    return min_epoch


df = pd.read_csv('Tweets.csv')
df = df.reindex(np.random.permutation(df.index))  
df = df[['text', 'airline_sentiment']]
df.text = df.text.apply(remove_stopwords).apply(remove_mentions)

X_train, X_test, y_train, y_test = train_test_split(df.text, df.airline_sentiment, test_size=0.1, random_state=37)

tk = Tokenizer(num_words=NB_WORDS,
               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{"}~\t\n',
               lower=True,
               char_level=False,
               split=' ')
tk.fit_on_texts(X_train)

X_train_oh = tk.texts_to_matrix(X_train, mode='binary')
X_test_oh = tk.texts_to_matrix(X_test, mode='binary')

le = LabelEncoder()
y_train_le = le.fit_transform(y_train)
y_test_le = le.transform(y_test)
y_train_oh = to_categorical(y_train_le)
y_test_oh = to_categorical(y_test_le)




# conclusion - how a model wth more ids  can be less acc than a model with lesser ids 

print("************************Good MODEL*******************")
X_train_rest, X_valid, y_train_rest, y_valid = train_test_split(X_train_oh, y_train_oh, test_size=0.995, random_state=317)
#import pdb; pdb.set_trace()
overfit_model1 = models.Sequential()
overfit_model1.add(layers.Dense(8, activation='relu', input_shape=(NB_WORDS,)))
overfit_model1.add(layers.Dense(1024, activation='relu'))
overfit_model1.add(layers.Dense(3, activation='softmax'))
overfit_model1.name = 'uc3_perfect'
overfit_history1 = deep_model(overfit_model1, X_train_rest, y_train_rest, X_valid, y_valid)
overfit_min1 = optimal_epoch(overfit_history1)
eval_metric(overfit_model1, overfit_history1, 'loss')
eval_metric(overfit_model1, overfit_history1, 'accuracy')


print("************************Regular accurate MODEL*******************")
X_train_rest, X_valid, y_train_rest, y_valid = train_test_split(X_train_oh, y_train_oh, test_size=0.998, random_state=307)
reg_model = models.Sequential()
reg_model.add(layers.Dense(64, activation='relu', input_shape=(NB_WORDS,)))
#reduced_model.add(layers.Dropout(0.5))
reg_model.add(layers.Dense(3, activation='softmax'))
reg_history = deep_model(reg_model, X_train_rest, y_train_rest, X_valid, y_valid)
reg_min = optimal_epoch(reg_history)
eval_metric(reg_model, reg_history, 'loss')




print("************************Reduced inaccurate MODEL*******************")
X_train_rest, X_valid, y_train_rest, y_valid = train_test_split(X_train_oh, y_train_oh, test_size=0.995, random_state=307)
reduced_model = models.Sequential()
reduced_model.add(layers.Dense(12, activation='relu', input_shape=(NB_WORDS,)))
#reduced_model.add(layers.Dropout(0.5))
reduced_model.add(layers.Dense(3, activation='softmax'))
reduced_model.name = 'uc3_underfit'
reduced_history = deep_model(reduced_model, X_train_rest, y_train_rest, X_valid, y_valid)
reduced_min = optimal_epoch(reduced_history)
eval_metric(reduced_model, reduced_history, 'loss')


overfit1_results = test_model(overfit_model1, X_train_oh, y_train_oh, X_test_oh, y_test_oh, overfit_min1)
reduced_results = test_model(reduced_model, X_train_oh, y_train_oh, X_test_oh, y_test_oh, reduced_min)


#compare_models_by_metric(overfit_model, perfect_model, overfit_history, perfect_history, 'val_accuracy')
compare_models_by_metric(overfit_model1, reduced_model, overfit_history1, perfect_history, 'val_accuracy')
# compare_models_by_metric(base_model, reduced_model, base_history, reduced_history, 'val_accuracy')
# compare_models_by_metric(base_model,rereduced_model , base_history, rereduced_history, 'val_accuracy')
# compare_models_by_metric(reduced_model, rereduced_model, reduced_history, rereduced_history, 'val_accuracy')
