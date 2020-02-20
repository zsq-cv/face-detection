'''
生成关于人脸的正负样本
给出两个rect的iou计算函数
根据原始样本（正样本train.txt val.txt）获取任意比例的1:1的负样本，负样本与正样本的iou<0.3
把正负样本打乱，放入新的文件train3.txt val3.txt
'''
import torch
import numpy as np
import pandas as pd
import PIL.Image as Image
import os
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as transforms
import random
import math

def show_face_landmarks(path,img,rect,rect2=None,kpt=None):
    img1 = Image.open(os.path.join(os.path.join(path,'data'),img))
    _,h,w = transforms.ToTensor()(img1).size()
    h,w = h/100.0,w/100.0
    if (h>=12 or w>=12):
        h,w = h/2.0,w/2.0
    plt.figure(figsize=(h*2,w*2))
    plt.imshow(img1)
    plt.gca().add_patch(plt.Rectangle((rect[0], rect[1]), rect[2], rect[3], color='green', fill=False, linewidth=1))
    if rect2 is not None:
        plt.gca().add_patch(
            plt.Rectangle((rect2[0], rect2[1]), rect2[2], rect2[3], color='blue', fill=False, linewidth=1))
    if kpt is not None:
        plt.gca().scatter(kpt[:, 0] + rect[0], kpt[:, 1] + rect[1], marker='.')
    plt.show()

# compute iou
def compute_iou(rect1,rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    x2 = x2 - x1
    y2 = y2 - y1
    x2_2 = x2 + w2
    y2_2 = y2 + h2
    if (x2 <= 0 and x2_2 <= 0) or x2 - w1 >= 0:
        iou = 0
    elif (y2 <= 0 and y2_2 <= 0) or y2 - h1 >= 0:
        iou = 0
    else:
        w = min(w1, x2_2) - max(x2, 0)
        h = min(h1, y2_2) - max(y2, 0)
        # print('w:',w)
        # print('h:',h)
        iou = w * h / (w1 * h1 + w2 * h2 - w * h) * 1.0
    return iou

def rect_lessthan_iou(rect,rate=1.0):
    x, y, w, h = rect
    rate2 = 0
    rand_x = random.uniform(w * (1 - math.sqrt(2 * rate / (1 + rate))),
                            w * (1 - math.sqrt(2 * rate2 / (1 + rate2)))) * random.choice([-1, 1])
    rand_y = random.uniform(h * (1 - math.sqrt(2 * rate / (1 + rate))),
                            h * (1 - math.sqrt(2 * rate2 / (1 + rate2)))) * random.choice([-1, 1])
    x2 = x + rand_x
    y2 = y + rand_y
    return x2, y2, w, h

###rectangle iou: 0
def rect_iou0(rect):
    x, y, w, h = rect
    rate2 = 0.0
    # rand_x = random.uniform(w * (1 - math.sqrt(2 * rate2 / (1 + rate2))), w * (1 + 1e-1)) * random.choice([-1, 1])
    # rand_y = random.uniform(h * (1 - math.sqrt(2 * rate2 / (1 + rate2))), h * (1 + 1e-1)) * random.choice([-1, 1])
    rand_x = random.uniform(w, w * (1 + 1e-1)) * random.choice([-1, 1])
    rand_y = random.uniform(h, h * (1 + 1e-1)) * random.choice([-1, 1])
    return x + rand_x, y + rand_y, w, h

def generate_sample_neg(path, sample_txt, data_path, iou_rate, sample_num):
    '''

    :param path: samples path
    :param sample_txt: sample txt name, its content that img_name,rect,landmarks
    :param data_path: samples path
    :param iou_rate: this iou rate is bigger then the iou of the generate negative sample and original(postive) sample
    :param sample_num: sample number
    :return: negative samples, it include that img_name,rect
    '''

    if sample_num == 0:
        return None
    train_list = open(os.path.join(path, sample_txt))

    ####image:image:img_name(I/001.jpg),rect:rectangle[x,y,w,h],rects:[rect1,rect2....],all rect in sample image
    samples = {'image': [], 'rect': [], 'rects': []}

    while True:
        line = train_list.readline()
        if not line:
            break
        # count += 1
        item_dict = line.split()
        image, rect = item_dict[0], np.array(item_dict[1:5]).astype('double')
        samples['image'].append(image)
        samples['rect'].append(rect)

    for i in range(len(samples['image'])):
        rects = []
        for j in range(len(samples['image'])):
            # print(samples['image'][i] == samples['image'][j])
            if (samples['image'][i] == samples['image'][j]):
                rects.append(list(samples['rect'][j]))
        samples['rects'].append(rects)

    samples_neg = {'image': [], 'rect': []}

    i = 0
    count = 0
    while i < len(samples['image']):
        # for i in range(len(samples['image'])):
        rect = samples['rect'][i]
        if (iou_rate == 0.0):
            rect_neg = rect_iou0(rect)
            # print('rect_neg:', rect_neg)
        else:
            rect_neg = rect_lessthan_iou(rect, rate=iou_rate)
        rect_neg = list(rect_neg)

        ####if (x,y)<0 or (x2,y2)>(weight,height), it need crop
        img = Image.open(os.path.join(data_path, samples['image'][i]))
        weight, height = img.size
        # print('w h:', weight, height)
        x, y, x2, y2 = rect_neg[0], rect_neg[1], rect_neg[0] + rect_neg[2], rect_neg[1] + rect_neg[3]

        if (x < 0):
            rect_neg[0] = 0
            rect_neg[2] = x2
        if (y < 0):
            rect_neg[1] = 0
            rect_neg[3] = y2

        if (x2 > weight):
            rect_neg[2] = weight - rect_neg[0]
        if (y2 > height):
            rect_neg[3] = height - rect_neg[1]

        #####if iou of the generate rect and others in same image is bigger iou_rate, we drop this sample
        rect_neg_iou = True
        for item in samples['rects'][i]:
            iou = compute_iou(rect_neg, item)
            #####two case:iou_rate:0.0,bigger than 0.0
            if (iou >= iou_rate and iou_rate > 0.0) or (iou > iou_rate and iou_rate == 0.0):
                rect_neg_iou = False
                break
        if (rect_neg_iou):
            # print('rect_neg2:', rect_neg)
            samples_neg['image'].append(samples['image'][i])
            samples_neg['rect'].append(rect_neg)
            count += 1
        if (count == sample_num):
            break
        if (i == len(samples['image']) - 1):
            i = 0
        else:
            i += 1

    print('total samples:{}'.format(len(samples['image'])))
    if (iou_rate == 0.0):
        print('generate negative samples(iou:{}):{}'.format(iou_rate, len(samples_neg['image'])))
    else:
        print('generate negative samples(0<iou:<{}):{}'.format(iou_rate, len(samples_neg['image'])))

    return samples_neg


def generate_data2(path, data_path, filename, sample_txt, iou_rate, pos_neg_rate=0.7, iou0_proportion=0.01):
    '''

    :param path: samples path
    :param data_path: samples path
    :param filename: postive sample file name
    :param sample_txt: sample txt name, its content that img_name,rect,landmarks
    :param iou_rate: this iou rate is bigger then the iou of the generate negative sample and original(postive) sample
    :param pos_neg_rate: proportion of postive samples to total samples
    :param iou0_proportion: proportion of samples(iou=0.0)
    :return:
    '''
    train_list = open(os.path.join(path, sample_txt))

    samples = {'image': [], 'rect': [], 'landmarks': []}

    while True:
        line = train_list.readline()
        if not line:
            break
        # count += 1
        item_dict = line.split()
        image, rect, landmarks = item_dict[0], np.array(item_dict[1:5]).astype('double'), np.array(
            item_dict[5:])
        samples['image'].append(image)
        samples['rect'].append(rect)
        samples['landmarks'].append(landmarks)

    ####negative sample number
    samples_neg_num = int(len(samples['image']) * (1 - pos_neg_rate) / pos_neg_rate)
    ####negative sample number(iou=0.0)
    samples_neg_num_iou0 = int(samples_neg_num * iou0_proportion)
    ####negative sample number(iou>0.0 and iou<iou_rate)
    samples_neg_num_iou_other = samples_neg_num - samples_neg_num_iou0

#     print(samples_neg_num, samples_neg_num_iou0, samples_neg_num_iou_other)
    samples_neg_num_0 = generate_sample_neg(path, sample_txt, data_path, 0.0, samples_neg_num_iou0)
    samples_neg_num_other = generate_sample_neg(path, sample_txt, data_path, iou_rate, samples_neg_num_iou_other)

    pos_i = 0
    neg_i_0 = 0
    neg_i_other = 0
    samples_list = []
    if samples is not None:
        len_pos = len(samples['image'])
    else:
        len_pos = 0

    len_pos = len(samples['image']) if samples is not None else 0
    len_neg_0 = len(samples_neg_num_0['image']) if samples_neg_num_0 is not None else 0
    len_neg_other = len(samples_neg_num_other['image']) if samples_neg_num_other is not None else 0

    while True:
        # np.random.seed(0)
        choice = np.random.choice([0, 1, 2], p=[pos_neg_rate, (1 - pos_neg_rate) * iou0_proportion,
                                                (1 - pos_neg_rate) * (1 - iou0_proportion)])
        ##postive sample
        line = ''
        if choice == 0 and pos_i < len_pos:
            # print(choice, end='')
            line = samples['image'][pos_i] + ' ' + ' '.join(samples['rect'][pos_i].astype('str')) + ' ' + ' '.join(
                samples['landmarks'][pos_i].astype('str')) + ' 1'
            pos_i += 1
        elif choice == 1 and neg_i_0 < len_neg_0:
            # print(choice, end='')
            line = samples_neg_num_0['image'][neg_i_0] + ' ' + ' '.join(
                np.array(samples_neg_num_0['rect'][neg_i_0]).astype('str')) + ' ' + '0'
            neg_i_0 += 1
        elif choice == 2 and neg_i_other < len_neg_other:
            # print(choice, end='')
            line = samples_neg_num_other['image'][neg_i_other] + ' ' + ' '.join(np.array(samples_neg_num_other['rect'][
                                                                                             neg_i_other]).astype(
                'str')) + ' ' + '0'
            neg_i_other += 1
        elif pos_i == len_pos and neg_i_0 == len_neg_0 and neg_i_other == len_neg_other:
            break
        if (line != ''):
            samples_list.append(line + '\n')

    print('total samples:{}'.format(len(samples_list)))
    print('postive samples:{}'.format(pos_i))
    print('negative samples(iou=0.0):{}'.format(neg_i_0))
    print('negative samples(0<iou<{}):{}'.format(iou_rate, neg_i_other))

    file_train = open(os.path.join(path, filename), 'w')
    file_train.writelines(samples_list)
    file_train.close()
    print('this file location:{}'.format(os.path.join(path, filename)))
    print('generate data is finish!')


#### show image
def show_image(path, filename, id):
    train_list = open(os.path.join(path, filename))

    count = 0
    while True:
        line = train_list.readline()
        if not line or count == id:
            break
        count += 1

    sample_array = line.split()
    landmarks = None
    if len(sample_array) > 6:
        image, rect, landmarks = sample_array[0], np.array(sample_array[1:5]).astype('double'), np.array(
            sample_array[5:-1]).reshape(-1, 2).astype('double')
    else:
        image, rect = sample_array[0], np.array(sample_array[1:5]).astype('double')
    print('img:', image)
    print('rect:', rect)
    show_face_landmarks(path, image, rect, kpt=landmarks)


def main():
    path = './'
    # generate_data(path + 'data', path + '/train2.txt', path + '/val2.txt',expand_rate=0.25)

    #     generate_data2(path, path, 'train3.txt', 'train.txt', 0.3, 0.7, 0.01)
    generate_data2(path, path, 'train3.txt', 'train.txt', 0.3, 0.5, 0.1)
    generate_data2(path, path, 'val3.txt', 'val.txt', 0.3, 0.5, 0.1)
    # show_image(path ,'train3.txt',10)


if __name__ == '__main__':
    main()
