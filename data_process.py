'''
样本初始化
原始人脸图片框扩展0.25倍
对超过图像边框进行截取
关键点超过图像的样本被删除
分别生成正负样本比例为7:3
生成train.txt val.txt
'''
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import random
import os
import matplotlib.pyplot as plt
import cv2
import PIL.Image as Image

# expand size
def expand_rect(x,y,w,h,image_w,image_h,expand_rate):
    #     x_center=w/2.0
    #     y_center=h/2.0
    w_new = w * np.sqrt(1 + expand_rate)
    h_new = h * np.sqrt(1 + expand_rate)
    x_new = x + w / 2.0 - w_new / 2.0
    y_new = y + h / 2.0 - h_new / 2.0
    if (x_new < 0):
        x_new = 0
    if (y_new < 0):
        y_new = 0
    if (w_new > image_w - x_new):
        w_new = image_w - x_new
    if (h_new > image_h - y_new):
        h_new = image_h - y_new
    return x_new, y_new, w_new, h_new

# generate train and test
def generate_data(path,datas_path, train_path, val_path, expand_rate=0.0, train_val_rate=0.7):
    # count = 1
    num = 0
    train_num = 0
    val_num = 0

    file_train = open(train_path, 'w')
    # file_train.writelines()
    file_train.close()
    file_val = open(val_path, 'w')
    # file_val.writelines()
    file_val.close()

#     os.chdir(datas_path)
#     os.chdir(path)
    # print(os.listdir())
    list_path = os.listdir(datas_path)
    print(list_path)

    for item in list_path:

        item_num = 0
        item_train_num = 0
        item_val_num = 0

        data_path = os.path.join(datas_path,os.path.join(item, 'label.txt'))
        print(data_path)
        # thefile = open(data_path, 'rb')
        count = 0
        thefile = open(data_path)
        while True:
            buffer = thefile.read(1024 * 8192)
            if not buffer:
                break
            count += buffer.count('\n')
        thefile.close()
        print('this file {} rows: {}'.format(data_path, count))

        # data_list = open('./data/I/label.txt')
        data_list = open(data_path)

        train_list = []
        val_list = []
        while True:
            line = data_list.readline()
            if not line:
                break
            item_num += 1
            # every 100 times print
            if (item_num % 100 == 0):
                print('row: {}/{}, rate of progress:{:.2f}'.format(item_num, count, 1.0 * item_num / count))
            sample_array = line.split()
            image, rect, kpt = sample_array[0], np.array(sample_array[1:5]).astype('double'), np.array(
                sample_array[5:]).reshape(-1, 2).astype('double')
            rect[2] = rect[2] - rect[0]
            rect[3] = rect[3] - rect[1]
            # img = Image.open('./data/I/' + image)
            imagepath = os.path.join(datas_path,item)
            img = Image.open(os.path.join(imagepath, image))
            _, h, w = transforms.ToTensor()(img).size()
            rect = expand_rect(*rect, w, h, expand_rate)
            kpt = kpt - np.array(rect[:2])
            kpt_x_min = np.min(kpt[:, 0])
            kpt_x_max = np.max(kpt[:, 0])
            kpt_y_min = np.min(kpt[:, 1])
            kpt_y_max = np.max(kpt[:, 1])
            # print('rect:',rect[0])
            # print('kpt x y min max:{} {} {} {}'.format(kpt_x_min,kpt_y_min, kpt_x_max, kpt_y_max))

            if (kpt_x_min >= 0 and kpt_y_min >= 0 and kpt_x_max <= rect[2] and kpt_y_max <= rect[3]):
                line_new = image + ' ' + ' '.join(np.round(np.array(rect), 5).astype('str')) + ' ' + ' '.join(
                    np.round(kpt, 5).reshape(-1).astype('str'))

                rand = random.randint(0, 9)
                if (rand < train_val_rate * 10):
                    train_list.append(datas_path+'/'+item + '/' + line_new + '\n')
                    item_train_num += 1
                else:
                    val_list.append(datas_path+'/'+item + '/' + line_new + '\n')
                    item_val_num += 1
            # else:
            #     print('rect:', rect)
            #     print('kpt x y min max:{} {} {} {}'.format(kpt_x_min, kpt_y_min, kpt_x_max, kpt_y_max))

        file_train = open(train_path, 'a')
        file_train.writelines(train_list)
        file_train.close()
        file_val = open(val_path, 'a')
        file_val.writelines(val_list)
        file_val.close()
        data_list.close()
        train_num += item_train_num
        val_num += item_val_num
        num += item_num

    print('landmarks of train:{}'.format(train_num))
    print('landmarks of val:{}'.format(val_num))
    print('landmarks of drop:{}'.format(num - train_num - val_num))
    print('landmarks of all:{}'.format(num))
    print('these files is finish!')


def main():
    path = './'
    generate_data(path,'data',path + '/train.txt',path + '/val.txt',0.25,0.7)

if __name__ == '__main__':
    main()





