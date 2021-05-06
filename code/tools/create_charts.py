'''
input - file/path where logs are stored
task - create plots
'''

from os import listdir
import matplotlib.pyplot as plt_loss
import matplotlib.pyplot as plt_acc
import pandas as pd

def drawPlot_acc_loss(model_list,model_name_list,file_path):
    model_count = len(model_list)
    linestyle_str = ['solid', 'dotted', 'dashed', 'dashdot']

    for i in range(model_count):
        model_name = model_name_list[i]
        train_loss = model_list[i].history['loss']
        val_loss = model_list[i].history['val_loss']
        epochs = range(0, len(val_loss))
        plt_loss.plot(epochs, train_loss, color='red', label=model_name + 'Train Loss', linestyle=linestyle_str[i], linewidth=i + 1)
        plt_loss.plot(epochs, val_loss, color='green', label=model_name + 'val loss', linestyle=linestyle_str[i], linewidth=i + 1)

    plt_loss.title('Training and Validation loss')
    plt_loss.xlabel('Epochs')
    plt_loss.ylabel('Loss')
    plt_loss.legend()
    plt_loss.savefig(file_path + "/Loss.png")
    plt_loss.clf()
    plt_loss.close()

    for i in range(model_count):
        model_name = model_name_list[i]
        train_acc = model_list[i].history['accuracy']
        val_acc = model_list[i].history['val_accuracy']
        epochs = range(0, len(val_acc))
        plt_acc.plot(epochs, train_acc, color='red', label=model_name + 'Train Acc', linestyle=linestyle_str[i], linewidth=i + 1)
        plt_acc.plot(epochs, val_acc, color='green', label=model_name + 'val Acc', linestyle=linestyle_str[i], linewidth=i + 1)

    plt_acc.title('Training and Validation Acc')
    plt_acc.xlabel('Epochs')
    plt_acc.ylabel('Accuracy')
    plt_acc.legend()
    plt_acc.savefig(file_path + "/Acc.png")
    plt_acc.clf()
    plt_acc.close()
