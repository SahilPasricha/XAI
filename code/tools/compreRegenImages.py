from os import listdir
import matplotlib.pyplot as plt
import random

def writeRgeneImages(input_images,regen_orig_images,regen_pred_images,count,path):

    plt.figure(figsize=(30, 5))
    for i in range(count):
        random_image = random.randint(0, input_images.shape[0]-1)
        ax = plt.subplot(3,count,i+1)
        plt.imshow(input_images[random_image])
        plt.title("original_"+str(random_image))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction original
        ax = plt.subplot(3,count,count+i+1)
        plt.imshow(regen_orig_images[random_image])
        plt.title("reconstructed_"+str(random_image))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction after regression
        ax = plt.subplot(3, count, count*2 + i + 1)
        plt.imshow(regen_pred_images[random_image])
        plt.title("reg. rec...ed_"+str(random_image))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    plt.savefig(path + "/ae-images.png")

