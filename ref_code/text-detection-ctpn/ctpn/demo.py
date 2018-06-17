from __future__ import print_function
import tensorflow as tf
import numpy as np
import os, sys, cv2
import glob
import shutil
import time
import matplotlib.pyplot as plt


sys.path.append(os.getcwd())
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg,cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.utils.timer import Timer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg

os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def IOU_2Box(gt_box, predict_box):
    """
    自定义函数，计算两矩形 IOU，传入为均为矩形对角线，（x,y）  坐标。
    """
    x1 = gt_box[0]
    y1 = gt_box[1]
    width1 = gt_box[2] - gt_box[0]
    height1 = gt_box[3] - gt_box[1]

    x2 = predict_box[0]
    y2 = predict_box[1]
    width2 = predict_box[2] - predict_box[0]
    height2 = predict_box[3] - predict_box[1]

    endx = max(x1+width1,x2+width2)
    startx = min(x1,x2)
    width = width1+width2-(endx-startx)

    endy = max(y1+height1,y2+height2)
    starty = min(y1,y2)
    height = height1+height2-(endy-starty)

    if width <=0 or height <= 0:
        ratio = 0 # 重叠率为 0
    else:
        Area = width*height # 两矩形相交面积
        Area1 = width1*height1
        Area2 = width2*height2
        ratio = Area*1./(Area1+Area2-Area)
    # return IOU
    return ratio


def IOU_mapBoexs(img, gt_boxes, predict_boxes):
    '''
    采用与、或的方法计算整张图的IOU
    box 格式为：x1,y1,x2,y2
    '''
    img_size = img.shape  # 读进来的是H*W (即y*x)
    gt_map = np.zeros((img_size[1], img_size[0]),dtype=np.uint8)
    predict_map = np.zeros((img_size[1], img_size[0]),dtype=np.uint8)
    for gt_box in gt_boxes:
        gt_map[gt_box[0]:gt_box[2] + 1, gt_box[1]:gt_box[3] + 1] = True
    for predict_box in predict_boxes:
        predict_map[predict_box[0]:predict_box[2] + 1, predict_box[1]:predict_box[3] + 1] = True
    I_val = np.sum(np.sum(np.logical_and(gt_map, predict_map), axis=0))
    U_val = np.sum(np.sum(np.logical_or(gt_map, predict_map), axis=0))

    if U_val <= 1e-16:
        print('IOU_mapBoexs 计算出现错误:被除数==0')
        return -1.0

    iou = I_val/U_val
    if iou > 1.0:
        print('IOU_mapBoexs 计算出现错误:iou>1.0')
        return -1.0
    return iou


def Texts_2_Boxes(file_content, format):
    boxes = []
    if format == 'HW_label':
        for line in file_content:
            splitted_line = line.strip().lower().split(' ')
            xmin = int(splitted_line[2])
            ymin = int(splitted_line[3])
            xmax = int(splitted_line[4])
            ymax = int(splitted_line[5])
            boxes.append([xmin, ymin, xmax, ymax])
    elif format == 'CTPN_result':
        for line in file_content:
            splitted_line = line.strip().lower().split(' ')
            xmin = int(splitted_line[0])
            ymin = int(splitted_line[1])
            xmax = int(splitted_line[2])
            ymax = int(splitted_line[3])
            boxes.append([xmin, ymin, xmax, ymax])
    boxes = np.array(boxes)
    return boxes



def calculate_IOU(cfg, test_sample_img_loc,test_sample_split_label_loc,output_dir_name):
    iou_list = []
    iou_file = open(output_dir_name + 'map_iou.txt', 'w')  # 用来记录整张图的iou的文件
    gt_file_names = os.listdir(os.path.join(test_sample_split_label_loc))
    gt_file_names.sort()
    predict_file_names = os.listdir(os.path.join(output_dir_name, 'split_label/'))
    predict_file_names.sort()
    for gt_file_name in gt_file_names:
        _, basename = os.path.split(gt_file_name)
        stem, ext = os.path.splitext(basename)
        stem = stem.split('_')[-1]  # 获取image的真实名称
        predict_file_name = 'res_' + stem + '.txt'
        # chech the existance of predict_file_name
        if predict_file_name not in predict_file_names:
            print('prediction Missing: ' + predict_file_name)
            continue
        # check the existance of
        gt_img_name = os.path.join(test_sample_img_loc, stem + '.jpg')
        if not os.path.exists(gt_img_name):
            print('image Missing: ' + gt_img_name)
            continue

        # read gt file
        gt_file = open(os.path.join(test_sample_split_label_loc, basename), 'r')
        gt_content = gt_file.readlines()
        gt_file.close()
        # read prediction file
        predict_file = open(os.path.join(output_dir_name, 'split_label/', predict_file_name), 'r')
        predict_content = predict_file.readlines()
        predict_file.close()
        # read image
        img = cv2.imread(gt_img_name)
        gt_boxes = Texts_2_Boxes(gt_content, 'HW_label')
        predict_boxes = Texts_2_Boxes(predict_content, 'CTPN_result')
        iou = IOU_mapBoexs(img, gt_boxes, predict_boxes)
        iou_list.append(iou)
        print('image ' + stem + '.jpg' + ' iou: %.4f' % iou)
        iou_file.write(stem + '.jpg' + ' ' + '%.4f' % iou + '\n')

    iou_file.close()
    return iou_list



def output_settings(begin_time_tag, path_list):
    begin_time_tag = begin_time_tag
    test_sample_loc = path_list[0]
    test_sample_img_loc = path_list[1]
    test_sample_split_label_loc = path_list[2]
    output_dir_name = path_list[3]
    output_boxed_image_loc = path_list[4]
    output_split_label_loc = path_list[5]

    f = open(output_dir_name + 'path_settings.txt', 'w')
    f.write('begin_time_tag = ' + begin_time_tag + '\n')
    f.write('test_sample_loc = ' + test_sample_loc + '\n')
    f.write('---> test_sample_img_loc = ' + test_sample_img_loc + '\n')
    f.write('---> test_sample_split_label_loc = ' + test_sample_split_label_loc + '\n')
    f.write('output_dir_name = ' + output_dir_name + '\n')
    f.write('---> output_boxed_image_loc = ' + output_boxed_image_loc + '\n')
    f.write('---> output_split_label_loc = ' + output_split_label_loc + '\n')
    # output text_connect_cfg
    f.write('-----------------------------------------\n')
    f.write('text_connect_cfg.SCALE:%d' % TextLineCfg.SCALE + '\n')
    f.write('text_connect_cfg.MAX_SCALE:%d' % TextLineCfg.MAX_SCALE + '\n')
    f.write('text_connect_cfg.TEXT_PROPOSALS_WIDTH:%d' % TextLineCfg.TEXT_PROPOSALS_WIDTH + '\n')
    f.write('text_connect_cfg.MIN_NUM_PROPOSALS:%d' % TextLineCfg.MIN_NUM_PROPOSALS + '\n')
    f.write('text_connect_cfg.MIN_RATIO:%.2f' % TextLineCfg.MIN_RATIO + '\n')
    f.write('text_connect_cfg.LINE_MIN_SCORE:%.4f' % TextLineCfg.LINE_MIN_SCORE + '\n')
    f.write('text_connect_cfg.MAX_HORIZONTAL_GAP:%d' % TextLineCfg.MAX_HORIZONTAL_GAP + '\n')
    f.write('text_connect_cfg.TEXT_PROPOSALS_MIN_SCORE:%.4f' % TextLineCfg.TEXT_PROPOSALS_MIN_SCORE + '\n')
    f.write('text_connect_cfg.TEXT_PROPOSALS_NMS_THRESH:%.4f' % TextLineCfg.TEXT_PROPOSALS_NMS_THRESH + '\n')
    f.write('text_connect_cfg.MIN_V_OVERLAPS:%.4f' % TextLineCfg.MIN_V_OVERLAPS + '\n')
    f.write('text_connect_cfg.MIN_SIZE_SIM:%.4f' % TextLineCfg.MIN_SIZE_SIM + '\n')
    f.close()

    # 复制训练时的yml文件
    shutil.copyfile('text.yml', output_dir_name + 'text.yml')


def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f



def draw_boxes(img, image_name, boxes, scale, output_dir_name):
    base_name = image_name.split('/')[-1]
    with open(output_dir_name + 'split_label/' + 'res_{}.txt'.format(base_name.split('.')[0]), 'w') as f:
        for box in boxes:
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            if box[8] >= 0.9:
                color = (0, 255, 0)
            elif box[8] >= 0.8:
                color = (255, 0, 0)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
            cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

            min_x = min(int(box[0]/scale),int(box[2]/scale),int(box[4]/scale),int(box[6]/scale))
            min_y = min(int(box[1]/scale),int(box[3]/scale),int(box[5]/scale),int(box[7]/scale))
            max_x = max(int(box[0]/scale),int(box[2]/scale),int(box[4]/scale),int(box[6]/scale))
            max_y = max(int(box[1]/scale),int(box[3]/scale),int(box[5]/scale),int(box[7]/scale))

            line = ' '.join([str(min_x),str(min_y),str(max_x),str(max_y)])+'\r\n'
            f.write(line)

    img=cv2.resize(img, None, None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_LINEAR)



def ctpn(sess, net, image_name, output_dir_name):
    timer = Timer()
    timer.tic()

    img = cv2.imread(image_name)
    img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    scores, boxes = test_ctpn(sess, net, img)

    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    draw_boxes(img, image_name, boxes, scale, output_dir_name=output_dir_name)
    base_name = image_name.split('/')[-1]
    cv2.imwrite(os.path.join(output_dir_name, "boxed_image/", base_name.split('.')[0] + '.jpg'), img)
    timer.toc()
    print(('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0]))
    return timer.total_time



if __name__ == '__main__':
    ######## test sample name ########
    # test_sample_loc = '/home/crown/WORK_space/PRHW-BigProject/ref_code/text-detection-ctpn/data/test_small/'
    # test_sample_loc = '/home/crown/WORK_space/PRHW-BigProject/ref_code/text-detection-ctpn/data/demo/'
    # test_sample_loc = '/home/crown/WORK_space/PRHW-BigProject/ID_dataset_train/'
    test_sample_loc = '/home/crown/WORK_space/PRHW-BigProject/ID_dataset_valid/'


    test_sample_img_loc = test_sample_loc + 'image/'
    test_sample_split_label_loc = test_sample_loc +'split_label/'

    begin_time_tag = time.strftime('%m-%d_%H:%M:%S', time.localtime(time.time()))
    output_dir_name = '../data/results/'
    if os.path.exists(output_dir_name):
        shutil.rmtree(output_dir_name)
    os.makedirs(output_dir_name)
    os.makedirs(output_dir_name + "boxed_image/")
    os.makedirs(output_dir_name + "split_label/")

    ######## 输出实验的配置 #######
    output_settings(begin_time_tag, [test_sample_loc, test_sample_img_loc, test_sample_split_label_loc, output_dir_name,
                     output_dir_name + "boxed_image/", output_dir_name + "split_label/"])

    ######## 开始实验 #######
    # read config file
    cfg_from_file('text.yml')

    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # load network
    net = get_network("VGGnet_test")
    # load model
    print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    saver = tf.train.Saver()

    try:
        ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in range(2):
        _, _ = test_ctpn(sess, net, im)

    im_names = glob.glob(os.path.join(test_sample_img_loc, '*.png')) + \
               glob.glob(os.path.join(test_sample_img_loc, '*.jpg'))

    # begin text detection
    time_list = []
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(('Demo for {:s}'.format(im_name)))
        single_time = ctpn(sess, net, im_name, output_dir_name=output_dir_name)
        time_list.append(single_time)


    # output calcualtion time
    time_file = open(output_dir_name + 'calculation_time.txt', 'w')
    for i in range(len(time_list)):
        time_file.write("%.4f" % time_list[i] + ' s' + '\n')
    time_file.close()


    # calculate IOU-map
    print('====================== calculate IOU of each image ====================')
    IOU_list = calculate_IOU(cfg, test_sample_img_loc, test_sample_split_label_loc, output_dir_name)
    # draw iou result
    mean_iou_list = np.mean(IOU_list) * np.ones(np.shape(IOU_list))
    plt.figure()
    plt.plot(IOU_list, linewidth=2)
    plt.plot(mean_iou_list, linewidth=2)
    plt.title(test_sample_loc + ' iou_curve')
    plt.savefig(output_dir_name + 'iou_curve.jpg', dpi=300)
