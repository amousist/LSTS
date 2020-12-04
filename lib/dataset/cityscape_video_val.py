# --------------------------------------------------------
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Copyright (c) 2017 ShanghaiTech PLUS Group
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Zheng Zhang
# --------------------------------------------------------

import cPickle
import os
import cv2
import numpy as np
import itertools

from imdb import IMDB
from PIL import Image

class Cityscape_Video_Val_Sequence(IMDB):
    def __init__(self, image_set, root_path, dataset_path, config, result_path=None):
        """
        fill basic information to initialize imdb
        :param image_set: leftImg8bit_train, etc
        :param root_path: 'selective_search_data' and 'cache'
        :param dataset_path: data and results
        :return: imdb object
        """
        image_set_main_folder, image_set_sub_folder = image_set.split('_', 1)
        super(Cityscape_Video_Val_Sequence, self).__init__('cityscape_video_val_sequence', image_set, root_path, dataset_path,
                                              result_path)  # set self.name

        self.image_set_main_folder = image_set_main_folder
        self.image_set_main_prefix = image_set_main_folder.split('Sequence')[0]
        self.image_set_sub_folder = image_set_sub_folder
        self.root_path = root_path
        self.data_path = dataset_path
        self.num_classes = 19
        self.image_set_index = self.load_image_set_index()
        self.video_rec_dict =self.load_image_set_rec_dict()
        self.num_images = len(self.image_set_index)
        print 'num_images', self.num_images
        self.config = {'comp_id': 'comp4',
                       'use_diff': False,
                       'min_size': 2}
        self.global_config = config

    def load_image_set_rec_dict(self):
        video_rec_dict = {}
        with open(os.path.join(self.root_path, 'cityscapes', 'cityscapes_leftImg8bit_sequence_index_dict.txt')) as f:
            for line in f.readlines():
                line = line.split()
                video_rec_dict[line[0]] = int(line[1])
        return video_rec_dict


    def load_image_set_index(self):
        """
        find out which indexes correspond to given image set
        :return: the indexes of given image set
        """
        # Collection all subfolders
        image_set_main_folder_path = os.path.join(self.data_path, self.image_set_main_folder, self.image_set_sub_folder)
        image_name_set = []
        with open(os.path.join(self.root_path, 'cityscapes', '%s_VideoSet.txt' % self.image_set_sub_folder)) as f:
            for line in f.readlines():
                image_name_set.append(line.strip())

        index_set = ['' for x in range(len(image_name_set))]
        valid_index_count = 0
        for i, image_name in enumerate(image_name_set):
            splited_name_set = image_name.split('_')
            ext_split = splited_name_set[len(splited_name_set) - 1].split('.')
            ext = ext_split[len(ext_split) - 1]
            if splited_name_set[len(splited_name_set) - 1] != 'flip.png' and ext == 'png':
                index_set[valid_index_count] = "_".join(splited_name_set[:len(splited_name_set) - 1])
                valid_index_count += 1

        return index_set[:valid_index_count]

    def image_path_from_index(self, index):
        """
        find the image path from given index
        :param index: the given index
        :return: the image path
        """
        index_folder = index.split('_')[0]
        image_file = os.path.join(self.data_path, self.image_set_main_folder, self.image_set_sub_folder, index_folder,
                                  index + '_' + self.image_set_main_prefix + '.png')
        return image_file

    def annotation_path_from_index(self, index):
        """
        find the gt path from given index
        :param index: the given index
        :return: the image path
        """
        index_folder = index.split('_')[0]
        image_file = os.path.join(self.data_path, 'gtFine', self.image_set_sub_folder, index_folder,
                                  index + '_gtFine_labelTrainIds.png')
        return image_file

    def load_segdb_from_index(self, index, flag):
        """
        load segdb from given index
        :param index: given index
        :return: segdb
        """
        seg_rec = dict()
        seg_rec['image'] = self.image_path_from_index(index)
        size = cv2.imread(seg_rec['image']).shape
        seg_rec['height'] = size[0]
        seg_rec['width'] = size[1]
        seg_rec['frame_name'] = index
        frame_20_index = int(index.split('_')[-1]) + 19 - flag
        frame_20_name = '_'.join(index.split('_')[:-1]) + '_%06d'%frame_20_index
        rec_20_frame_idx = self.video_rec_dict[frame_20_name]
        rec_idx = rec_20_frame_idx - (19 - flag)
        seg_rec['rec_idx'] = rec_idx
        seg_rec['frame_flag'] = flag

        if index in self.video_rec_dict:
            seg_rec['seg_cls_path'] = self.annotation_path_from_index(index)
        else:
            seg_rec['seg_cls_path'] = None

        seg_rec['flipped'] = False
        # add for video
        seg_rec['split'] = self.image_set_sub_folder  # train or val
        seg_rec['city'] = index.split('_')[0]
        seg_rec['seg_id'] = index.split('_')[1]
        return seg_rec

    def gt_segdb(self):
        """
        return ground truth image regions database
        :return: imdb[image_index]['', 'flipped']
        """
        print('seg roidb start')
        cache_file = os.path.join(self.cache_path, self.name + '_gt_VideoValSegdb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                segdb = cPickle.load(fid)
                print('{} gt segdb loaded from {}'.format(self.name, cache_file))
            return segdb
        print("create seg roidb first")
        gt_segdb = []
        flag = 0
        for i, index in enumerate(self.image_set_index):
            print('create %d/%d' % (i, len(self.image_set_index)))
            flag = flag % 30
            print('flag', flag)
            gt_segdb.append(self.load_segdb_from_index(index, flag))
            flag = flag + 1

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_segdb, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote gt segdb to {}'.format(cache_file))
        return gt_segdb

    def getpallete(self, num_cls):
        """
        this function is to get the colormap for visualizing the segmentation mask
        :param num_cls: the number of visulized class
        :return: the pallete
        """
        n = num_cls
        pallete_raw = np.zeros((n, 3)).astype('uint8')
        pallete = np.zeros((n, 3)).astype('uint8')

        pallete_raw[6, :] = [111, 74, 0]
        pallete_raw[7, :] = [81, 0, 81]
        pallete_raw[8, :] = [128, 64, 128]
        pallete_raw[9, :] = [244, 35, 232]
        pallete_raw[10, :] = [250, 170, 160]
        pallete_raw[11, :] = [230, 150, 140]
        pallete_raw[12, :] = [70, 70, 70]
        pallete_raw[13, :] = [102, 102, 156]
        pallete_raw[14, :] = [190, 153, 153]
        pallete_raw[15, :] = [180, 165, 180]
        pallete_raw[16, :] = [150, 100, 100]
        pallete_raw[17, :] = [150, 120, 90]
        pallete_raw[18, :] = [153, 153, 153]
        pallete_raw[19, :] = [153, 153, 153]
        pallete_raw[20, :] = [250, 170, 30]
        pallete_raw[21, :] = [220, 220, 0]
        pallete_raw[22, :] = [107, 142, 35]
        pallete_raw[23, :] = [152, 251, 152]
        pallete_raw[24, :] = [70, 130, 180]
        pallete_raw[25, :] = [220, 20, 60]
        pallete_raw[26, :] = [255, 0, 0]
        pallete_raw[27, :] = [0, 0, 142]
        pallete_raw[28, :] = [0, 0, 70]
        pallete_raw[29, :] = [0, 60, 100]
        pallete_raw[30, :] = [0, 0, 90]
        pallete_raw[31, :] = [0, 0, 110]
        pallete_raw[32, :] = [0, 80, 100]
        pallete_raw[33, :] = [0, 0, 230]
        pallete_raw[34, :] = [119, 11, 32]

        train2regular = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

        for i in range(len(train2regular)):
            pallete[i, :] = pallete_raw[train2regular[i] + 1, :]

        pallete = pallete.reshape(-1)

        return pallete

    def evaluate_segmentations(self, pred_segmentations=None):
        """
        top level evaluations
        :param pred_segmentations: the pred segmentation result
        :return: the evaluation results
        """
        if not (pred_segmentations is None):
            self.write_segmentation_result(pred_segmentations)

        info = self._py_evaluate_segmentation()
        return info

    def get_confusion_matrix(self, gt_label, pred_label, class_num):
        """
        Calcute the confusion matrix by given label and pred
        :param gt_label: the ground truth label
        :param pred_label: the pred label
        :param class_num: the nunber of class
        :return: the confusion matrix
        """
        index = (gt_label * class_num + pred_label).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((class_num, class_num))

        for i_label in range(class_num):
            for i_pred_label in range(class_num):
                cur_index = i_label * class_num + i_pred_label
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

        return confusion_matrix

    def _py_evaluate_segmentation(self):
        """
        This function is a wrapper to calculte the metrics for given pred_segmentation results
        :return: the evaluation metrics
        """
        dff_test_offset = self.global_config.TEST.OFFSET
        dff_test_epoch = self.global_config.TEST.test_epoch
        res_file_folder = os.path.join(self.result_path,
                                       'results_offset_{}_epoch_{}'.format(dff_test_offset, dff_test_epoch))

        confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        cache_file = os.path.join(self.cache_path, self.name + '_gt_segdb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                segdb = cPickle.load(fid)
        for i, index in enumerate(self.image_set_index):
            seg_gt_info = segdb[i]
            seg_gt = np.array(Image.open(seg_gt_info['seg_cls_path'])).astype('float32')

            seg_pathes = os.path.split(seg_gt_info['seg_cls_path'])
            res_image_name = seg_pathes[1][:-len('_gtFine_labelTrainIds.png')]
            res_subfolder_name = os.path.split(seg_pathes[0])[-1]
            res_save_folder = os.path.join(res_file_folder, res_subfolder_name)
            res_save_path = os.path.join(res_save_folder, res_image_name + '.png')

            seg_pred = np.array(Image.open(res_save_path)).astype('float32')

            seg_pred = cv2.resize(seg_pred, (seg_gt.shape[1], seg_gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            ignore_index = seg_gt != 255
            seg_gt = seg_gt[ignore_index]
            seg_pred = seg_pred[ignore_index]

            confusion_matrix += self.get_confusion_matrix(seg_gt, seg_pred, self.num_classes)

        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)

        IU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IU = IU_array.mean()

        return {'meanIU': mean_IU, 'IU_array': IU_array}

    def write_segmentation_result(self, segmentation_results):
        """
        Write the segmentation result to result_file_folder
        :param segmentation_results: the prediction result
        :param result_file_folder: the saving folder
        :return: [None]
        """
        dff_test_offset = self.global_config.TEST.OFFSET
        dff_test_epoch = self.global_config.TEST.test_epoch
        res_file_folder = os.path.join(self.result_path,
                                       'results_offset_{}_epoch_{}'.format(dff_test_offset, dff_test_epoch))

        if not os.path.exists(res_file_folder):
            os.mkdir(res_file_folder)

        pallete = self.getpallete(256)
        cache_file = os.path.join(self.cache_path, self.name + '_gt_segdb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                segdb = cPickle.load(fid)
        for i, index in enumerate(self.image_set_index):
            seg_gt_info = segdb[i]
            seg_pathes = os.path.split(seg_gt_info['seg_cls_path'])
            res_image_name = seg_pathes[1][:-len('_gtFine_labelTrainIds.png')]
            res_subfolder_name = os.path.split(seg_pathes[0])[-1]
            res_save_folder = os.path.join(res_file_folder, res_subfolder_name)
            res_save_path = os.path.join(res_save_folder, res_image_name + '.png')

            if not os.path.exists(res_save_folder):
                os.makedirs(res_save_folder)

            segmentation_result = np.uint8(np.squeeze(np.copy(segmentation_results[i])))
            segmentation_result = Image.fromarray(segmentation_result)
            segmentation_result.putpalette(pallete)
            segmentation_result.save(res_save_path)
