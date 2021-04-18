#from skimage.measure import structural_similarity as ssim
from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
import cv2
import numpy as np



def mse(imageO, imageB):
    err = np.sum((imageO.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageO.shape[0] * imageO.shape[1])
    return err

def compare_image(imageO, imageB):
    imageB = imageB.numpy()
    m = mse(imageO, imageB)
    s = measure.compare_ssim(imageO, imageB)
    return m,s


def compare_image_arrays(x_test, predicted_images_orig, predicted_images_reg, test_labels, plot_path,random_array):
    n_image = random_array.size
    plt.figure(figsize=(30, 8))
    
    for i in range(n_image):
        random_image = int(random_array[i])
        #comp_csv(predicted_images_orig[random_image],predicted_images_reg[random_image])
        #import pdb;pdb.set_trace()
        
        
        #setup the input image 
        ax = plt.subplot(5,n_image,i+1)
        plt.imshow(x_test[random_image])
        plt.title("Inp."+str(test_labels[random_image]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        #Image reconstructed using AE
        mse_prid, sim_prid = compare_image(x_test[random_image], predicted_images_orig[random_image])
        ax = plt.subplot(5,n_image,n_image+i+1)
        plt.imshow(predicted_images_orig[random_image])
        #plt.title("reconstructed_"+str(random_image))
        plt.title("rec.MSE: %.2f, SSIM: %.2f" % (mse_prid, sim_prid))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        #image reconstructed using ML and Regression 
        mse_reg, sim_reg = compare_image(x_test[random_image], predicted_images_reg[random_image])
        ax = plt.subplot(5, n_image, n_image*2 + i + 1)
        plt.imshow(predicted_images_reg[random_image])
        #plt.title("reg. rec...ed_"+str(random_image))
        plt.title("reg.MSE: %.2f, SSIM: %.2f" % (mse_reg, sim_reg))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        diff_image = calc_pixel_diff(predicted_images_orig[random_image],predicted_images_reg[random_image])
        ax = plt.subplot(5, n_image, n_image*3 + i + 1)
        plt.imshow(diff_image)
        #plt.title("reg. rec...ed_"+str(random_image))
        plt.title("Pred - Reg" )
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig(plot_path)   
        
        

def comp_csv(img0,imgR):
        import matplotlib.pyplot as plt1
        import matplotlib.pyplot as plt2
        plt1.imshow(imgR)
        plt2.imshow(imgR)
        plt1.savefig("1.png") 
        plt2.savefig("2.png")
        img1 = cv2.imread("1.png")
        img2 = cv2.imread("2.png")
        diff = cv2.absdiff(img1, img2)
        mask = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        th = 0.01
        imask = mask>th
        canvas = np.zeros_like(img2, np.uint8)
        canvas[imask] = img2[imask]
        cv2.imwrite("result.png", canvas)
        
def calc_pixel_diff(imgP, imgR):
    diffrence = imgP - imgR
    #diffrence[diffrence>10]= 255
    return diffrence

#image - image is just noice , either use boositng libarires or mask of main image , so that they compare only the area thats white in main image
