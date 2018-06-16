import os
import shutil
import numpy as np
import  cv2

def split_label_to_files(gt_path, gt_file_name, out_path):
    if not os.path.exists(out_path):
        print(out_path + ' ===== not exist, create one')
        os.makedirs(out_path)
    else:
        print(out_path + ' ===== already exist, I remove & new it')
        shutil.rmtree(out_path)
        os.makedirs(out_path)

    gt_file = os.path.join(gt_path, gt_file_name)
    with open(gt_file, 'r') as f:
        lines = f.readlines()

    last_im_base_name = ''
    last_file = ''
    for line in lines:
        splitted_line = line.strip().lower().split(' ')
        im_name = splitted_line[1]
        im_base_name = im_name.strip().lower().split('.')[0]

        if im_base_name != last_im_base_name:
            if last_file != '':
                last_file.close()
            print('create new file: ' + 'gt_' + im_base_name + '.txt')
            last_file = open(os.path.join(out_path, 'gt_' + im_base_name) + '.txt', 'a')
            last_im_base_name = im_base_name
            last_file.writelines(line)
        else:
            last_file.write(line)


def copy_img_by_split_plabel(img_path, split_label_path, out_path):
    if not os.path.exists(out_path):
        print(out_path + ' ===== not exist, create one')
        os.makedirs(out_path)
    else:
        print(out_path + ' ===== already exist, I remove & new it')
        shutil.rmtree(out_path)
        os.makedirs(out_path)
    file_names = os.listdir(split_label_path)
    file_names.sort()

    img_names = os.listdir(img_path)
    img_names.sort()

    for file_name in file_names:
        _, basename = os.path.split(file_name)
        stem, ext = os.path.splitext(basename)
        stem = stem.split('_')[-1]
        img_name_jpg = stem + '.jpg'
        img_name_png = stem + '.png'
        if img_name_jpg in img_names:
            img = cv2.imread(img_path + img_name_jpg)
            cv2.imwrite(out_path + img_name_jpg, img)
        elif img_name_png in img_names:
            img = cv2.imread(img_path + img_name_png)
            cv2.imwrite(out_path + img_name_png, img)
        else:
            print('Error: ' + img_path + img_name_jpg + ' not found in original imageset')



if __name__ == '__main__':
    img_path = '/home/crown/WORK_space/PRHW-BigProject/ID_dataset/' + 'image/'
    gt_path = '/home/crown/WORK_space/PRHW-BigProject/ID_dataset/' + 'label/'

    gt_file_name = 'train.txt'
    out_path = '/home/crown/WORK_space/PRHW-BigProject/ID_dataset_train/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    else:
        shutil.rmtree(out_path)
        os.makedirs(out_path)
    print('start split label')
    split_label_to_files(gt_path, gt_file_name, out_path + 'split_label/')
    print('end split, start copy img to output folder')
    copy_img_by_split_plabel(img_path, out_path + 'split_label/', out_path + 'image/')
    print('Finish img-copy')