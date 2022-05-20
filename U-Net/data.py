import os
import cv2

# 打开文件
path = "./CASIA1"
dirs = os.listdir(path)
print(dirs)  # 输出所有子文件和文件夹

for file in dirs:
    pic_dir = os.path.join(path, file)  # images中子文件夹的路径
    for i in os.listdir(pic_dir):
        image_dir = os.path.join(pic_dir, i)  # images中每个子文件夹中图片的路径
        img = cv2.imread(image_dir)

        print(image_dir)  # 输出图片的路径
        print(img)  # 输出图片

        # 读取
        img = cv2.imread(image_dir)
        # 显示
        cv2.imshow('window_title', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # 保存
        Img_Name = "./CASIA" + str(i)
        cv2.imwrite(Img_Name, img)