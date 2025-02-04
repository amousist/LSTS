# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Zhengkai Jiang
# --------------------------------------------------------
import argparse
import pprint
import logging
import time
import os
import numpy as np
import mxnet as mx

from symbols import *
from dataset import *
from core.loader import TestLoader_Impression_Online, TestLoader_Impression_Offline
from core.tester import Predictor, pred_eval, pred_eval_multiprocess, pred_eval_impression_online, pred_eval_multiprocess_impression_online, pred_eval_impression_offline, pred_eval_multiprocess_impression_offline
from utils.load_model import load_param

def get_predictor(sym, sym_instance, cfg, arg_params, aux_params, test_data, ctx):
    # infer shape
    data_shape_dict = dict(test_data.provide_data_single)
    sym_instance.infer_shape(data_shape_dict)
    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)

    # decide maximum shape
    data_names = [k[0] for k in test_data.provide_data_single]
    label_names = None
    max_data_shape = [[('data', (1, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES]))),
                       ('data_key', (1, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES]))),]]

    # create predictor
    predictor = Predictor(sym, data_names, label_names,
                          context=ctx, max_data_shapes=max_data_shape,
                          provide_data=test_data.provide_data, provide_label=test_data.provide_label,
                          arg_params=arg_params, aux_params=aux_params)
    return predictor

def get_predictor_impression_online(sym, sym_instance, cfg, arg_params, aux_params, test_data, ctx):
    # infer shape
    data_shape_dict = dict(test_data.provide_data_single)
    sym_instance.infer_shape(data_shape_dict)
    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)
    # decide maximum shape
    data_names = [k[0] for k in test_data.provide_data_single]
    label_names = None
    max_data_shape = [[('data_oldkey', (1, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES]))),
                       ('data_newkey', (1, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES]))),
                       ('data_cur', (1, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES]))),
		               ('impression', (1, 1024, 38, 63)),
		               ('key_feat_task', (1, 1024, 38, 63))]]
    # create predictor
    print 'provide_data', test_data.provide_data
    predictor = Predictor(sym, data_names, label_names,
                          context=ctx, max_data_shapes=max_data_shape,
                          provide_data=test_data.provide_data, provide_label=test_data.provide_label,
                          arg_params=arg_params, aux_params=aux_params)
    return predictor

def get_predictor_impression_offline(sym, sym_instance, cfg, arg_params, aux_params, test_data, ctx):
    # infer shape
    data_shape_dict = dict(test_data.provide_data_single)
    sym_instance.infer_shape(data_shape_dict)
    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)
    # decide maximum shape
    data_names = [k[0] for k in test_data.provide_data_single]
    label_names = None
    max_data_shape = [[('data_oldkey', (1, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES]))),
                       ('data_newkey', (1, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES]))),
                       ('data_cur', (1, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES]))),
		               ('impression', (1, 1024, 38, 63)),
		               ('key_feat_task', (1, 1024, 38, 63))]]
    # create predictor
    print 'provide_data', test_data.provide_data
    predictor = Predictor(sym, data_names, label_names,
                          context=ctx, max_data_shapes=max_data_shape,
                          provide_data=test_data.provide_data, provide_label=test_data.provide_label,
                          arg_params=arg_params, aux_params=aux_params)
    return predictor

def test_rcnn(cfg, dataset, image_set, root_path, dataset_path,
              ctx, prefix, epoch,
              vis, ignore_cache, shuffle, has_rpn, proposal, thresh, logger=None, output_path=None):
    if not logger:
        assert False, 'require a logger'
    # print cfg
    pprint.pprint(cfg)
    logger.info('testing cfg:{}\n'.format(pprint.pformat(cfg)))
    # load symbol and testing data
    key_sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
    cur_sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
    key_sym = key_sym_instance.get_key_test_symbol(cfg)
    cur_sym = cur_sym_instance.get_cur_test_symbol(cfg)
    imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
    roidb = imdb.gt_roidb()
    # get test data iter
    # split roidbs
    gpu_num = len(ctx)
    roidbs = [[] for x in range(gpu_num)]
    roidbs_seg_lens = np.zeros(gpu_num, dtype=np.int)
    for x in roidb:
        gpu_id = np.argmin(roidbs_seg_lens)
        roidbs[gpu_id].append(x)
        roidbs_seg_lens[gpu_id] += x['frame_seg_len']
    # get test data iter
    test_datas = [TestLoader(x, cfg, batch_size=1, shuffle=shuffle, has_rpn=has_rpn) for x in roidbs]
    # load model
    arg_params, aux_params = load_param(prefix, epoch, process=True)
    # create predictor
    key_predictors = [get_predictor(key_sym, key_sym_instance, cfg, arg_params, aux_params, test_datas[i], [ctx[i]]) for i in range(gpu_num)]
    cur_predictors = [get_predictor(cur_sym, cur_sym_instance, cfg, arg_params, aux_params, test_datas[i], [ctx[i]]) for i in range(gpu_num)]
    # start detection
    pred_eval_multiprocess(gpu_num, key_predictors, cur_predictors, test_datas, imdb, cfg, vis=vis, ignore_cache=ignore_cache, thresh=thresh, logger=logger)

def test_rcnn_impression_online(cfg, dataset, image_set, root_path, dataset_path, ctx, prefix, epoch, vis, ignore_cache, shuffle, has_rpn, proposal, thresh, logger=None, output_path=None):
    if not logger:
        assert False, 'require a logger'
    # print cfg
    pprint.pprint(cfg)
    logger.info('testing cfg:{}\n'.format(pprint.pformat(cfg)))
    # load symbol and testing data
    first_sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
    key_sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
    cur_sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()

    first_sym = first_sym_instance.get_first_test_symbol_impression(cfg)
    key_sym = key_sym_instance.get_key_test_symbol_impression(cfg)
    cur_sym = cur_sym_instance.get_cur_test_symbol_impression(cfg)

    imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
    roidb = imdb.gt_roidb()

    # get test data iter
    # split roidbs
    gpu_num = len(ctx)
    roidbs = [[] for x in range(gpu_num)]
    roidbs_seg_lens = np.zeros(gpu_num, dtype=np.int)
    for x in roidb:
        gpu_id = np.argmin(roidbs_seg_lens)
        roidbs[gpu_id].append(x)
        roidbs_seg_lens[gpu_id] += x['frame_seg_len']
    # get test data iter
    test_datas = [TestLoader_Impression_Online(x, cfg, batch_size=1, shuffle=shuffle, has_rpn=has_rpn,
                                                        from_rec=cfg.TEST.from_rec) for x in roidbs]
    # load model
    arg_params, aux_params = load_param(prefix, epoch, process=True)
    # create predictor
    first_predictors = [get_predictor_impression_online(first_sym, first_sym_instance, cfg, arg_params, aux_params, test_datas[i],
                                 [ctx[i]]) for i in range(gpu_num)]
    key_predictors = [get_predictor_impression_online(key_sym, key_sym_instance, cfg, arg_params, aux_params, test_datas[i],
                                 [ctx[i]]) for i in range(gpu_num)]
    cur_predictors = [get_predictor_impression_online(cur_sym, cur_sym_instance, cfg, arg_params, aux_params, test_datas[i], [ctx[i]]) for i
        in range(gpu_num)]
    # start detection
    pred_eval_multiprocess_impression_online(gpu_num, first_predictors, cur_predictors,
                                             key_predictors, test_datas, imdb, cfg, vis=vis, ignore_cache=ignore_cache,
                                             thresh=thresh, logger=logger)

def test_rcnn_impression_offline(cfg, dataset, image_set, root_path, dataset_path, ctx, prefix, epoch, vis, ignore_cache, shuffle, has_rpn, proposal, thresh, logger=None, output_path=None):
    if not logger:
        assert False, 'require a logger'
    # print cfg
    pprint.pprint(cfg)
    logger.info('testing cfg:{}\n'.format(pprint.pformat(cfg)))

    # load symbol and testing data
    first_sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
    key_sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
    cur_sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()

    first_sym = first_sym_instance.get_first_test_symbol_impression(cfg)
    key_sym = key_sym_instance.get_key_test_symbol_impression(cfg)
    cur_sym = cur_sym_instance.get_cur_test_symbol_impression(cfg)

    imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
    roidb = imdb.gt_roidb()
    # get test data iter
    # split roidbs
    gpu_num = len(ctx)
    roidbs = [[] for x in range(gpu_num)]
    roidbs_seg_lens = np.zeros(gpu_num, dtype=np.int)
    for x in roidb:
        gpu_id = np.argmin(roidbs_seg_lens)
        roidbs[gpu_id].append(x)
        roidbs_seg_lens[gpu_id] += x['frame_seg_len']
    # get test data iter
    test_datas = [TestLoader_Impression_Offline(x, cfg, batch_size=1, shuffle=shuffle, has_rpn=has_rpn,
                                                from_rec=cfg.TEST.from_rec) for x in roidbs]
    # load model
    arg_params, aux_params = load_param(prefix, epoch, process=True)
    # create predictor
    first_predictors = [get_predictor_impression_offline(first_sym, first_sym_instance, cfg, arg_params, aux_params, test_datas[i],
                                                         [ctx[i]]) for i in range(gpu_num)]
    key_predictors = [get_predictor_impression_offline(key_sym, key_sym_instance, cfg, arg_params, aux_params, test_datas[i],
                                                       [ctx[i]]) for i in range(gpu_num)]
    cur_predictors = [get_predictor_impression_offline(cur_sym, cur_sym_instance, cfg, arg_params, aux_params, test_datas[i], 
                                                       [ctx[i]]) for i in range(gpu_num)]
    # start detection
    pred_eval_multiprocess_impression_offline(gpu_num, first_predictors, cur_predictors,
                                              key_predictors, test_datas, imdb, cfg, vis=vis, ignore_cache=ignore_cache,
                                              thresh=thresh, logger=logger)

