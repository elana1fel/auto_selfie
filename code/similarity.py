import os
import itertools
import cv2
import numpy as np
import selfie_utils
from skimage.metrics import structural_similarity as compare_ssim
import matplotlib.pyplot as plt

def show_imgs(imgs_list, imgs_name_list=None, params=None, save=False):
    '''
    This func gets images and plots them side by side
    Input:
        imgs_list       np.array list       list of imgs to plot
        imgs_name_list  list of str         list of imgs name
        params          str                 str to describe running params to be used as title    
    '''
    imgs_num = len(imgs_list)
    if imgs_num == 4:
        fig, axarr = plt.subplots(2, 2)
        i_range = 2
        j_range = 2
    
        counter = 0
        for i in range(i_range):
            for j in range(j_range):
                axarr[i][j].imshow(imgs_list[counter], cmap='gray')
                if imgs_name_list is not None:
                    axarr[i][j].set_title(imgs_name_list[counter])
                counter += 1

    else:
        fig, axarr = plt.subplots(1, imgs_num)
        i_range = imgs_num

        for i in range(i_range):
            axarr[i].imshow(imgs_list[i], cmap='gray')
            if imgs_name_list is not None:
                axarr[i].set_title(imgs_name_list[i])

    if params is not None:
        fig.suptitle(params)

    save =True
    if save:
        params = params.split('\n')[0]
        plt.savefig("plots/" + params + ".png")

    else:
        plt.show()
    plt.close()


def mse(img1, img2):
	# computet Mean Squared Error between the two images
	err = np.sum((img1 - img2) ** 2)
	err /= float(img1.shape[0] * img2.shape[1])
	
	return err


def test_similarity(img1, img2, method=None, debug=False):
    '''
    This func checks whether two images are similiar

    Input:
        img1        np.array
        img2        np.array
        method      str         method for similarity test. it can be MSE or ssim
          
    Output:
        similarity  bool        whether two images are similar
    '''
    
    #check
    ssim_sim = True
    mse_sim = True

    (score, diff) = compare_ssim(img1, img2, full=True)
    if score < 0.75:
        ssim_sim = False
    err = mse(img1, img2)
    if err > 50.0:
        mse_sim = False

    if debug:
        show_imgs([img1, img2], params=f"ssim score is: {score:.2f} and MSE err is: {err:.2f}")
    return np.logical_and(ssim_sim, mse_sim)

if __name__ == "__main__":
    #input_path = r'test_video.mp4'
    # run(input_path)

    similarity_dataset_path = r'databases/similarity'
    files = os.listdir(similarity_dataset_path)
    for pair in itertools.combinations(files, r=2):
        print(f"comparing between {pair}")

        img1 = os.path.join(similarity_dataset_path, pair[0])
        img2 = os.path.join(similarity_dataset_path, pair[1])

        img1 = cv2.imread(img1)
        gray_img1, resized_frame1 = selfie_utils.edit_img(img1)

        img2 = cv2.imread(img2)
        gray_img2, resized_frame2 = selfie_utils.edit_img(img2)

        test_similarity(gray_img1, gray_img2, debug=True)

