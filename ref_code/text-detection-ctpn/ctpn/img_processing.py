import numpy as np
import os, sys, cv2


if __name__ == '__main__':
    project_path = '/home/crown/WORK_space/PRHW-BigProject/ref_code/text-detection-ctpn/'
    data_path = project_path + 'data/NetCommander_sampleData/' + 'image/'
    out_path = project_path + 'data/NC_processedData/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_path = out_path + 'image/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    files = os.listdir(data_path)
    files.sort()

    for file in files:
        _, basename = os.path.split(file)
        if basename.lower().split('.')[-1] not in ['jpg', 'png']:
            continue
        img_path = os.path.join(data_path, file)
        stem, ext = os.path.splitext(basename)
        print(img_path)
        img = cv2.imread(img_path)
        img_size = img.shape  # 读进来的是H*W (即y*x)

        # img = img[int(img_size[0]*7/8):-1, int(img_size[1]/2):-1, :]
        img = img[:, int(img_size[1]/2):-1, :]

        # padding操作
        BLACK = [0,0,0]
        h_padding_length = int(img_size[0]/100)
        w_padding_length = int(img_size[1]/100)
        padding_length = min(h_padding_length,w_padding_length)
        img = cv2.copyMakeBorder(img, padding_length, padding_length, padding_length, padding_length, \
                                 cv2.BORDER_CONSTANT, value=BLACK)

        # 二值化
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转灰度
        # _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

        # 展示图片
        # cv2.imshow('hello',img)
        # cv2.waitKey(0)

        re_name = os.path.join(out_path, stem) + '.jpg'
        cv2.imwrite(re_name, img)






