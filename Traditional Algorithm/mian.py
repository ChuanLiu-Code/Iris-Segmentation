import cv2
import math
import numpy as np
#import Image
#from PIL import Image
img = cv2.imread('test1.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#灰度图像 
#cv2.imshow('gray',gray)
#cv2.waitKey(0)
#hough transform
'''
def mean_binarization(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold = np.mean(img_gray)
    img_gray[img_gray>threshold] = 255
    img_gray[img_gray<=threshold] = 0
    plt.imshow(img_gray, cmap='gray')
    plt.show()
    return img_gray
    '''
circles1 = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,
100,param1=100,param2=30,minRadius=54,maxRadius=300)
cv2.imshow('gray2',circles1)
print("内环数据如下")
print(circles1)
circles = circles1[0,:,:]#提取为二维
circles = np.uint16(np.around(circles))#四舍五入，取整
print("内环数据取整如下")
print(circles)

#center_coordinates=(254,204)
#image = cv2.circle(img, center_coordinates, 77, (255,0,0), 2)
#cv2.imshow("image",image)
#cv2.imwrite("image.png",image)
for i in circles[:]: 
    cv2.circle(img,(i[0],i[1]),i[2],(255,255,255),-1)#画圆
    #cv2.circle(img,(i[0],i[1]),2,(0,0,0),10)#画圆心
    x=i[0]
    y=i[1]
    r=i[2]
#cv2.imshow('in circle',img)
#cv2.waitKey(0)
print('内环圆心')
print(x,y)
print("-------------------------------------------------")
############################################
gray1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#灰度图像
circles2 = cv2.HoughCircles(gray1,cv2.HOUGH_GRADIENT,1,
100,param1=100,param2=30,minRadius=56,maxRadius=200)
print("外环数据如下")
print(circles2)
circles=circles2[0,:,:]
print(circles)
#cir[y,x,r]=circles2[0,:,:]
circles=np.uint16(np.around(circles))
print(circles)
#image2=cv2.imread("image.png")
#image2 = cv2.circle(image2, (300,228), 107, (255,0,0), 2)
#cv2.imshow('out circle ds',image2)
#cv2.waitKey(0)
for i in circles[:]: 
    cv2.circle(img,(i[0],i[1]),i[2],(255,255,255),1)#画圆
    #cv2.circle(img,(i[0],i[1]),2,(0,0,0),10)#画圆心
    x1=i[0]
    y1=i[1]
    r1=i[2]
cv2.imshow('out circle',img)
cv2.waitKey(0)
print('外环圆心')
print(x1,y1)
#print(circles)
print("-------------------------------------------------")
"""
x=circles[0]
y=circles[1]
r=circles[2]
print(circles[0])
"""
#########################################
[height,width,pixels] = img.shape
print("图片尺寸如下")
print("高——宽————像素点")
print(height,width,pixels)
#########################################
print("-------------------------------------------------")

for i in range(width):
    for j in range(height):
        dis1=np.sqrt(np.square(i-x1)+np.square(j-y1))
        #print(dis1)
        if(dis1>r1):
            cv2.circle(img,(i,j),2,(255,255,255),-1)#画圆心
cv2.imwrite("out_a.png",img)
cv2.imshow('out around',img)
cv2.waitKey(0)

###########################################归一化##########################
###输入：内圆坐标(x,y),半径r
gama_in=1#归一化上边缘范围  一般半径的1.4倍较好
gama_out=2.2#归一化下边缘范围  一般半径的2.7倍较好
R_in = np.round(r*gama_in)
R_out = np.round(r*gama_out)
N=np.uint16(np.round(math.pi*(R_in+R_out)))
N=640
print(N)
M=np.uint16(np.round(R_out-R_in))
print("转换成16位后的值")
print(M)
alpha=2*math.pi/N
m1=[]
for i in range(1):
    for j in range(N):
        m1.append(x+R_in*math.cos(j*alpha))
m2=[]
for i in range(1):
    for j in range(N):
        m2.append(y+R_in*math.sin(j*alpha))
m3=[]
for i in range(1):
    for j in range(N):
        m3.append(x+R_out*math.cos(j*alpha))
m4=[]
for i in range(1):
    for j in range(N):
        m4.append(y+R_out*math.sin(j*alpha))
#m=np.mat(m)
#m1=np.mat(m1)
#m2=np.mat(m2)
#m3=np.mat(m3)
#q1=m2-m
#q2=m3-m1
img_rec=cv2.imread("out_a.png")
s1=[]
for i in range(M):
    for j in range(N):
        #s1=np.round(m+q1*i/M)
        s1.append(int(np.round(x+R_in*math.cos(j*alpha)+
                               (R_out-R_in)*math.cos(j*alpha)*i/M)))
        #s1=np.mat(s1)
#print(s1)
s1=np.mat(s1)
s1=s1.reshape(M,N)
print("s1-------------------")
print(s1)
s2=[]
for i in range(M):
    for j in range(N):
        #s2=np.round(m+q2*i/M)
        s2.append(int(np.round(x+R_in*math.cos(j*alpha)+
                               (R_out-R_in)*math.sin(j*alpha)*i/M)))
        #img_new=im.covert(s1(i,j),s2(i,j))
s2=np.mat(s2)
s2=s2.reshape(M,N)
print("s[1,1]--------------")
print(s1[1,1])
img_after = np.zeros([229,719,3],np.uint8)+255

for i in range(M):
    for j in range(N-M):
        x_end=s1[i,j]
        y_end=s2[i,j]
        img_after[i,j]=img_rec[x_end,y_end]
print("-------------------------------------------------")
print("M的值")
print(M)
print("N的值")
print(N)
cv2.imshow('Finish',img_after)
cv2.waitKey(0)
