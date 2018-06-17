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
            continue    # 没找到的话，就跳过这个图片的gt
        else:
            lines = f.readlines()   # 找到的话，就读取

        for line in lines:
            # 替换为助教给出的min 和 max，并进行缩放
            splitted_line = line.strip().lower().split(' ')
            xmin_original = float(splitted_line[2])
            ymin_original = float(splitted_line[3])
            xmax_original = float(splitted_line[4])
            ymax_original = float(splitted_line[5])

            class_flag = np.zeros((3,1),dtype=np.uint8)
            class_flag[0] = np.round(float(splitted_line[6]))     # 是否数字
            class_flag[1] = np.round(float(splitted_line[7]))     # 是否英文
            class_flag[2] = np.round(float(splitted_line[8]))     # 是否中文

            class_inx = 0
            if class_flag[0] > 0:
                class_inx += 1
            if class_flag[1] > 0:
                class_inx += 3
            if class_flag[2] > 0:
                class_inx += 5

            if class_inx == 1:  real_class = 0      # 数字
            elif class_inx==3:  real_class = 1      # 英文
            elif class_inx==5:  real_class = 2      # 汉字
            elif class_inx==4:  real_class = 3      # 数字+英文
            elif class_inx==6:  real_class = 4      # 数字+汉字
            elif class_inx==8:  real_class = 5      # 英文+汉字
            elif class_inx==9:  real_class = 6      # 数字+英文+汉字
            else:
                print('=============== 类别数据读取有误 =============== ')



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

            # 直接输出

            if not os.path.exists('with_class_label'):
                os.makedirs('with_class_label')
            with open(os.path.join('with_class_label', stem) + '.txt', 'a') as f:
                # f.writelines("text\t")
                f.writelines(str(int(xmin)))
                f.writelines("\t")
                f.writelines(str(int(ymin)))
                f.writelines("\t")
                f.writelines(str(int(xmax)))
                f.writelines("\t")
                f.writelines(str(int(ymax)))
                f.writelines("\t")
                f.writelines(str(int(real_class)))
                f.writelines("\n")




        ####  save resize picture
        re_name = os.path.join(out_path, stem) + '.jpg'
        # 在图上面表出方框，检验读取数据是否正确
        # draw_boxes(re_im, re_name, boxes)

        # 保存缩放后的图像
        cv2.imwrite(re_name, re_im)




