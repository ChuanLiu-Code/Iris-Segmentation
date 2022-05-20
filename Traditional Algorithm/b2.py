import cv2
#from image_gray.image_gray_methods import gray_mean_rgb
import numpy as np
#全局固定阈值
 
#需要给定的参数为阈值、阈值类型标签号和图片地址
#thresh:阈值
#flags：阈值类型标签号，0——cv2.THRESH_BINARY，1——cv2.THRESH_BINARY_INV，2——cv2.THRESH_TRUNC,
# 3——cv2.THRESH_TOZERO,4——cv2.THRESH_TOZERO_INV
def threshold_demo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   ##要二值化图像，必须先将图像转为灰度图
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print("threshold value %s" % ret)  #打印阈值，超过阈值显示为白色，低于该阈值显示为黑色
    cv2.imwrite("img_th.jpg",binary)
    cv2.imshow("threshold", binary) #显示二值化图像


def custom_demo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    m = np.reshape(gray, [1, h*w]) #将图像转为1行h*w列
    mean = m.sum() / (h*w)  #计算图像的均值，用均值作为阈值，来分割图像
    ret, binary = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
    print("threshold value %s" % ret)
    cv2.imwrite("img_th.jpg",binary)
    cv2.imshow("cudtom_binary", binary)


 
if __name__ == '__main__':
   image=cv2.imread("test1.jpg")
   image = cv2.GaussianBlur(image, (3,3), 0)
   threshold_demo(image)
   #custom_demo(image)

