import os
import shutil
import numpy as np





if __name__ == '__main__':
    gt_path = '/home/crown/WORK_space/PRHW-BigProject/ID_dataset/' + 'label'
    out_path = '/home/crown/WORK_space/PRHW-BigProject/ID_dataset/' + 'split_label'
    if not os.path.exists(out_path):
        print(out_path + ' ===== not exist')
        os.makedirs(out_path)
    else:
        print(out_path + ' ===== already exist, I remove & new it')
        shutil.rmtree(out_path)
        os.makedirs(out_path)

    gt_file = os.path.join(gt_path, 'train.txt')
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



