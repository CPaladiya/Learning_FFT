import cv2
import numpy as np
import matplotlib.pyplot as plt

#def plot_a_line(imRaw):

def count_vertical_stripes(image_path):
    # Step 1: Import required libraries
    # NumPy for numerical operations, Matplotlib for plotting, OpenCV for image processing
    
    # Step 2: Load input image
    img = cv2.imread(image_path)
    
    # Step 3: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayRaw = gray[50:51, :]

    # smoothing the image

    # Works all right
    # kernel = np.ones((5,5), np.float32)/50
    # dst = cv2.filter2D(gray,-1,kernel)

    # works better than the last one
    # kernel = np.ones((9,9), np.uint8)
    # dst = cv2.erode(gray, kernel, iterations = 1)
    # dst = cv2.dilate(gray, kernel, iterations = 1)

    # works quite well
    # dst = cv2.GaussianBlur(gray,(5,5),0)
    dst = gray
    
    dstRaw = dst[54:55, :]

    RangeOf = grayRaw.shape[1]
    #RangeOf = 100

    plt.subplot(321)
    plt.imshow(gray)
    plt.subplot(322)
    plt.imshow(dst)
    plt.subplot(323)
    plt.plot(np.arange(0,RangeOf,1),grayRaw[0][0:RangeOf],'r')
    plt.subplot(324)
    plt.plot(np.arange(0,RangeOf,1),dstRaw[0][0:RangeOf],'b')

    # Step 4: Apply 1D Fast Fourier Transform
    f = np.fft.fft(dstRaw)
    
    # Step 5: Take absolute value of complex output
    f_abs = np.abs(f)
    
    plt.subplot(325)
    plt.plot(np.arange(0,int(RangeOf/2),1),f_abs[0][0:int(RangeOf/2)],'b')
    plt.show()
    
    # # Step 9: Count number of vertical stripes
    max_indices = np.argmax(f_abs[:,2:-2], axis=1) + 2
    print("Maximum index : ", max_indices)
    return

image_path = 'TestImages/im5.jpg'
count_vertical_stripes(image_path)
# count = count_vertical_stripes(image_path)
# print('Number of vertical stripes:', count)
