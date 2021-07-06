import os
import cv2
import numpy as np


def search_mean_std(imgpath):

    list_imgs = search_images(imgpath)

    list_mean = []
    list_std  = []

    for idx, imgfile in enumerate(list_imgs):
        print(idx, imgfile)
        
        img = cv2.imread(imgfile, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        
        if len(img.shape) < 2: #remove gray color
            continue
    
        list_mean.append(np.mean(img, axis=(0, 1)))
        list_std.append(np.std(img, axis=(0, 1)))

    mean_var = np.mean(list_mean, axis=0)
    std_var  = np.mean(list_std,  axis=0)

    with open('mean_std.txt', 'w') as f:
        f.write(str(mean_var))
        f.write(str(std_var))

    f.close()

    return

def search_images(imgpath):

    list_imgs = []

    for dirpath, dirname, filename in os.walk(imgpath):

        if len(filename) > 0:
            for one in filename:
                name, fmt = one.split('.')
                file_path = os.path.join(dirpath, one)

                if os.path.isfile(file_path):
                    if fmt in ['jpg', 'JPG', 'png', 'PNG']:
                        list_imgs.append(file_path)

    print("total images : {:}".format(len(list_imgs)))

    return list_imgs




if __name__ == "__main__":
    #- neg, part, pos
    imgpath = "/datasets/WIDER_FACE_MASK/train_data/train_pnet/"

    search_mean_std(imgpath)
