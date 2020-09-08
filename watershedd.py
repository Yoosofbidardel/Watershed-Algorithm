import cv2
import matplotlib.pyplot as plt
import numpy as np



def display(img):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, 'gray')
    plt.show()
    
coin=cv2.imread("C:/pennies.jpg")
coind=cv2.medianBlur(coin,35)
gray_coin=cv2.cvtColor(coind,cv2.COLOR_BGR2GRAY)
ret ,thresh = cv2.threshold(gray_coin,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# erod = cv2.erode(thresh,np.ones((5,5)),iterations=5)
dist_trans=cv2.distanceTransform(thresh,cv2.DIST_L2,5)
rett ,dist=cv2.threshold(dist_trans,0.72*dist_trans.max(),255,0)
dist=np.uint8(dist)
unknown=thresh-dist
ret1 ,label=cv2.connectedComponents(dist)
label+=1
label[unknown==255]=0
label=cv2.watershed(coin,label)
contour,hierarchy = cv2.findContours(label,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
for i in range(len(contour)):
    if hierarchy[0][i][3]==-1:
        cv2.drawContours(coin,contour,i,(255,0,0),10)
display(coin)
