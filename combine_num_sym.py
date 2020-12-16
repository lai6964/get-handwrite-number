import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
# from my_aug import my_aug_up


pic_save_dir = './train/data/'
txt_save_dir = './train/label/'
symbols_label_path = 'simple_symbol.txt'
number_label_path = 'number_label.txt'
generate_number = 10

if not os.path.exists(pic_save_dir):
    os.mkdir("train")
    os.mkdir(pic_save_dir)
    os.mkdir(txt_save_dir)

num_imgs = []
sym_imgs = []
numbers = []
symbols = []
with open(symbols_label_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        words = line.split('\t')
        sym_imgs.append('./symbol_data/' + words[0])
        symbols.append(words[-1].rstrip('\n'))
symbol_total = len(symbols)-1

with open(number_label_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        words = line.split('\t')
        num_imgs.append('./number_data/' + words[0])
        numbers.append(words[-1].rstrip('\n'))
number_total = len(numbers)-1



for pic_num in range(generate_number):
    length = random.randint(0, 10)
    index = random.randint(0, number_total)
    img = cv2.imread(num_imgs[index])
    img = 255-img
    label = numbers[index]
    for i in range(length):
        if random.random() > 0.7:
            # concatenate symbol & number
            index = random.randint(0, symbol_total)
            img_tmp = cv2.imread(sym_imgs[index])
            w = img_tmp.shape[1]
            cut_len = random.randint(int(w / 4), int(w / 2))
            cut_tmp = img_tmp[:, cut_len:]
            add_tmp = img_tmp[:, :cut_len]
            ww = img.shape[1]
            #         img[:,ww-cut_len:] = cv2.addWeighted(img[:,ww-cut_len:],0.5, add_tmp,0.5,0)
            #         img[:,ww-cut_len:] = cv2.add(img[:,ww-cut_len:], add_tmp)
            for i in range(cut_len):
                for j in range(img.shape[0]):
                    img[j, ww - cut_len + i, 0] = min(add_tmp[j, i, 0], img[j, ww - cut_len + i, 0])
                    img[j, ww - cut_len + i, 1] = min(add_tmp[j, i, 1], img[j, ww - cut_len + i, 1])
                    img[j, ww - cut_len + i, 2] = min(add_tmp[j, i, 2], img[j, ww - cut_len + i, 2])

            label_tmp = symbols[index]
            if label_tmp =='\\times':###有的符号要用\转义
                label_tmp ='\\\\times'
            img = np.concatenate([img, cut_tmp], axis=1)
            label = label +' '+ label_tmp

        # concatenate number
        index = random.randint(0, number_total)
        img_tmp = cv2.imread(num_imgs[index])
        img_tmp = 255-img_tmp
        w = img_tmp.shape[1]
        cut_len = random.randint(int(w / 4), int(w / 2))
        cut_tmp = img_tmp[:, cut_len:]
        add_tmp = img_tmp[:, :cut_len]
        ww = img.shape[1]
        #     img[:,ww-cut_len:] = cv2.addWeighted(img[:,ww-cut_len:],0.5, add_tmp,0.5,0)
        #     img[:,ww-cut_len:] = cv2.add(img[:,ww-cut_len:], add_tmp)
        for i in range(cut_len):
            for j in range(img.shape[0]):
                img[j, ww - cut_len + i, 0] = min(add_tmp[j, i, 0], img[j, ww - cut_len + i, 0])
                img[j, ww - cut_len + i, 1] = min(add_tmp[j, i, 1], img[j, ww - cut_len + i, 1])
                img[j, ww - cut_len + i, 2] = min(add_tmp[j, i, 2], img[j, ww - cut_len + i, 2])
        label_tmp = numbers[index]
        img = np.concatenate([img, cut_tmp], axis=1)
        label = label +' '+ label_tmp
    # plt.imshow(img)
    # print(label)

    ######this for augment , sometimes should be rewrite again
    # img[img<255]=0
    # img=my_aug_up(img)

    ######### change color
    # if random.random()>0.7:
    #     h, w, c = img.shape
    #     r = random.randint(0, 255)
    #     g = random.randint(0, 255)
    #     b = random.randint(0, 255)
    #     for hh in range(h):
    #         for ww in range(w):
    #             if img[hh, ww, 0] < 255:
    #                 img[hh, ww, 0] = r  # int((255-img[hh,ww,0])*r)
    #                 img[hh, ww, 1] = g  # int((255-img[hh,ww,0])*g)
    #                 img[hh, ww, 2] = b  # int((255-img[hh,ww,0])*b)
    cv2.imwrite(pic_save_dir+str(pic_num)+'.png',img)
    with open(txt_save_dir+str(pic_num)+'.txt','w') as file:
        file.write(str(pic_num)+'.png'+'\t'+label)
    if pic_num%1000==0:
        print(pic_num)