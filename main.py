import cv2 as cv
import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt

# Calculate the OTSU's threshold
def find_otsu(img):
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    hist_norm = hist.ravel()/hist.sum()
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1
    for i in range(1,256):
        p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
        if q1 < 1.e-6 or q2 < 1.e-6:
            continue
        b1,b2 = np.hsplit(bins,[i]) # weights
        # finding means and variances
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    return thresh

def threshold(img, threshold):
    # using the global threshold to binary the img
    m, n = len(img),len(img[0])
    new_img = np.zeros((m,n))
    for x in range(0, len(img)):
        for y in range(0, len(img[0])):
            if img[x][y]>threshold:
                new_img[x][y]=255
    return new_img

# Use threshold methods to get the binary images
def get_bimg(P, img_p):
    img_path = os.path.join(P, img_p)
    print(img_path)
    img = cv.imread(img_path,0)
    # global thresholding 127
    th1 = threshold(img, 127)
    # Otsu's thresholding
    Otsu = find_otsu(img)
    th2 = threshold(img, Otsu)
    # Otsu's thresholding after Gaussian filtering
    blur = cv.GaussianBlur(img,(3,3),0)
    blur_Otsu = find_otsu(img)
    th3 = threshold(blur, blur_Otsu)
    # plot all the images and their histograms
    
    return blur, th1, th2, th3

# Define the 8 neighbours of the selected pixel
#   P9 P2 P3
#   P8 P1 P4
#   P7 P6 P5
def neighbours(x, y, image):
    '''Return 8-neighbours of point p1 of picture, in order'''
    i = image
    x1, y1, x_1, y_1 = x + 1, y - 1, x - 1, y + 1
    # print ((x,y))
    return [i[y1][x], i[y1][x1], i[y][x1], i[y_1][x1],  # P2,P3,P4,P5
            i[y_1][x], i[y_1][x_1], i[y][x_1], i[y1][x_1]]  # P6,P7,P8,P9

# Calculate the number of 0-1 transtions in the ordered equence P2, P3, ... P9
def transitions(neighbours):
    n = neighbours + neighbours[0:1]  # n = P2, ... P9, P2
    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:])) # zip(n, n[1:]) = [(P2, P3), (P3, P4), ... (P9, P2)]

# Main step of Thining
def Two_steps(image):
    step1 = step2 = [(-1, -1)]
    # Repeat the steps until there is no more changing
    while step1 or step2:
        # Step 1  label all points fitting all the conditions
        step1 = []
        for y in range(1, len(image) - 1):
            for x in range(1, len(image[0]) - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, image)
                if (image[y][x] == 1 and  # (Condition 0)
                        P4 * P6 * P8 == 0 and  # Condition 4
                        P2 * P4 * P6 == 0 and  # Condition 3
                        transitions(n) == 1 and  # Condition 2
                        2 <= sum(n) <= 6):  # Condition 1
                    step1.append((x, y))
        # Delete all points in step1
        for x, y in step1: image[y][x] = 0
        # Step 2 label all points fitting all the conditions
        step2 = []
        for y in range(1, len(image) - 1):
            for x in range(1, len(image[0]) - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, image)
                if (image[y][x] == 1 and  # (Condition 0)
                        P2 * P6 * P8 == 0 and  # Condition 4
                        P2 * P4 * P8 == 0 and  # Condition 3
                        transitions(n) == 1 and  # Condition 2
                        2 <= sum(n) <= 6):  # Condition 1
                    step2.append((x, y))
        # Delete all points in step2
        for x, y in step2: image[y][x] = 0
    return image

# Use the thin algorithm
def thin(img):
    a = np.array(img)
    for i in range(np.size(a, 0)):
        for j in range(np.size(a, 1)):
            if a[i][j] == 0:
                a[i][j] = 1
            else:
                a[i][j] = 0
    after = Two_steps(a)
    for i in range(np.size(after, 0)):
        for j in range(np.size(after, 1)):
            if after[i][j] == 1:
                after[i][j] = 0
            else:
                after[i][j] = 255
    # new_img = cv.fromarray(after)
    return after

if __name__ == '__main__':
    P = 'imges'
    A = []
    B = []
    for img_p in os.listdir(P):
        blur, th1, th2, th3 = get_bimg(P, img_p)
        img_path = os.path.join(P, img_p)
        img = cv.imread(img_path,0)

        images = [img, 0, th1, img, 0, th2, blur, 0, th3]
        thins = [thin(th1), thin(th2), thin(th3)]
        titles = ['Original Image','Histogram','Threshold=127','Thinning',
                'Original Image','Histogram',"Threshold={}".format(find_otsu(img)),'Thinning',
                'Filtered Image','Histogram',"Threshold={}".format(find_otsu(blur)),'Thinning']
        for i in range(3):
            plt.subplot(3,4,i*4+1),plt.imshow(images[i*3],'gray')
            plt.title(titles[i*4]), plt.xticks([]), plt.yticks([])
            plt.subplot(3,4,i*4+2),plt.hist(images[i*3].ravel(),256)
            plt.title(titles[i*4+1]), plt.xticks([]), plt.yticks([])
            plt.subplot(3,4,i*4+3),plt.imshow(images[i*3+2],'gray')
            plt.title(titles[i*4+2]), plt.xticks([]), plt.yticks([])
            plt.subplot(3,4,i*4+4),plt.imshow(thins[i],'gray')
            plt.title(titles[i*4+3]), plt.xticks([]), plt.yticks([])
        plt.savefig('Results/Result_{}.jpg'.format(img_p.split('.')[0]))
        plt.close()