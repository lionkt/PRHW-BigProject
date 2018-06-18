import os
import numpy as np
import math
import cv2 as cv2



########################
def draw_boxes(img,image_name,boxes):
    for box in boxes:
        color = (255, 0, 0)
        cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
        cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
        cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)
        # draw point
        cv2.circle(img,(int(box[0]), int(box[1])),radius=10,color=(0,0,255))    # r 画出来(xmin, ymin)
        cv2.circle(img,(int(box[6]), int(box[7])),radius=10,color=(0,255,0))    # g 画出来(xmax, ymax)

    # cv2.imwrite(image_name, img)



if __name__ == '__main__':
    path = '/home/crown/WORK_space/PRHW-BigProject/ID_dataset_train/' + 'image'
    gt_path = '/home/crown/WORK_space/PRHW-BigProject/ID_dataset_train/' + 'split_label'
    out_path = 're_image'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    files = os.listdir(path)
    files.sort()
    # files=files[0:10]   # 先测试小批数据

    for file in files:
        _, basename = os.path.split(file)
        if basename.lower().split('.')[-1] not in ['jpg', 'png']:
            continue
        stem, ext = os.path.splitext(basename)
        gt_file = os.path.join(gt_path, 'gt_' + stem + '.txt')
        img_path = os.path.join(path, file)
        print(img_path)
        img = cv2.imread(img_path)
        img_size = img.shape    # 读进来的是H*W (即y*x)
        im_size_min = np.min(img_size[0:2])
        im_size_max = np.max(img_size[0:2])

        im_scale = float(600) / float(im_size_min)
        if np.round(im_scale * im_size_max) > 1200:
            im_scale = float(1200) / float(im_size_max)
        re_im = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        re_size = re_im.shape

        boxes = []

        try:
            f = open(gt_file, 'r')
        except:
            continue    # 没找到gt_file的话，就跳过这个图片的gt
        else:
            lines = f.readlines()   # 找到的话，就读取

        for line in lines:
            # splitted_line = line.strip().lower().split(',')
            # pt_x = np.zeros((4, 1))
            # pt_y = np.zeros((4, 1))
            # pt_x[0, 0] = int(float(splitted_line[0]) / img_size[1] * re_size[1])
            # pt_y[0, 0] = int(float(splitted_line[1]) / img_size[0] * re_size[0])
            # pt_x[1, 0] = int(float(splitted_line[2]) / img_size[1] * re_size[1])
            # pt_y[1, 0] = int(float(splitted_line[3]) / img_size[0] * re_size[0])
            # pt_x[2, 0] = int(float(splitted_line[4]) / img_size[1] * re_size[1])
            # pt_y[2, 0] = int(float(splitted_line[5]) / img_size[0] * re_size[0])
            # pt_x[3, 0] = int(float(splitted_line[6]) / img_size[1] * re_size[1])
            # pt_y[3, 0] = int(float(splitted_line[7]) / img_size[0] * re_size[0])
            #
            # ind_x = np.argsort(pt_x, axis=0)
            # pt_x = pt_x[ind_x]
            # pt_y = pt_y[ind_x]
            #
            # if pt_y[0] < pt_y[1]:
            #     pt1 = (pt_x[0], pt_y[0])
            #     pt3 = (pt_x[1], pt_y[1])
            # else:
            #     pt1 = (pt_x[1], pt_y[1])
            #     pt3 = (pt_x[0], pt_y[0])
            #
            # if pt_y[2] < pt_y[3]:
            #     pt2 = (pt_x[2], pt_y[2])
            #     pt4 = (pt_x[3], pt_y[3])
            # else:
            #     pt2 = (pt_x[3], pt_y[3])
            #     pt4 = (pt_x[2], pt_y[2])
            #
            # xmin = int(min(pt1[0], pt2[0]))
            # ymin = int(min(pt1[1], pt2[1]))
            # xmax = int(max(pt2[0], pt4[0]))
            # ymax = int(max(pt3[1], pt4[1]))


            # 替换为助教给出的min 和 max，并进行缩放
            splitted_line = line.strip().lower().split(' ')
            xmin_original = float(splitted_line[2])
            ymin_original = float(splitted_line[3])
            xmax_original = float(splitted_line[4])
            ymax_original = float(splitted_line[5])

            xmin = int(xmin_original / img_size[1] * re_size[1])
            ymin = int(ymin_original / img_size[0] * re_size[0])
            xmax = int(xmax_original / img_size[1] * re_size[1])
            ymax = int(ymax_original / img_size[0] * re_size[0])


            if xmin < 0:
                xmin = 0
            if xmax > re_size[1] - 1:
                xmax = re_size[1] - 1
            if ymin < 0:
                ymin = 0
            if ymax > re_size[0] - 1:
                ymax = re_size[0] - 1

            width = xmax - xmin
            height = ymax - ymin

            # add min-length detection,借鉴了模式识别课
            if width <= 5 or height <=5:
                continue

            box = [xmin, ymin, xmax, ymin, xmin, ymax, xmax, ymax]
            boxes.append(box)

            # reimplement
            step = 10.0 #16.0
            x_left = []
            x_right = []
            x_left.append(xmin)
            x_left_start = int(math.ceil(xmin / step) * step)
            if x_left_start == xmin:
                x_left_start = xmin + step
            for i in np.arange(x_left_start, xmax, step):
                x_left.append(i)
            x_left = np.array(x_left)

            x_right.append(x_left_start - 1)
            for i in range(1, len(x_left) - 1):
                x_right.append(x_left[i] + step - 1)
            x_right.append(xmax)
            x_right = np.array(x_right)

            idx = np.where(x_left == x_right)
            x_left = np.delete(x_left, idx, axis=0)
            x_right = np.delete(x_right, idx, axis=0)

            if not os.path.exists('label_tmp'):
                os.makedirs('label_tmp')
            with open(os.path.join('label_tmp', stem) + '.txt', 'a') as f:
                for i in range(len(x_left)):
                    f.writelines("text\t")
                    f.writelines(str(int(x_left[i])))
                    f.writelines("\t")
                    f.writelines(str(int(ymin)))
                    f.writelines("\t")
                    f.writelines(str(int(x_right[i])))
                    f.writelines("\t")
                    f.writelines(str(int(ymax)))
                    f.writelines("\n")


        ####  save resize picture
        re_name = os.path.join(out_path, stem) + '.jpg'
        # 在图上面表出方框，检验读取数据是否正确
        # draw_boxes(re_im, re_name, boxes)

        # 保存缩放后的图像
        cv2.imwrite(re_name, re_im)




