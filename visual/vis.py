'''
This file provides 
visualization operations on market and duke dataset
'''

import sys

sys.path.append('..')

import argparse
import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets
import matplotlib
# matplotlib.use('agg')
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import os

import torch
import torch.nn as nn
import torch.optim as optim
from bisect import bisect_right
import os
from collections import OrderedDict
from visual import meter


# class Vis:
# 	def resume_from_model(self, model_path):
# 		'''resume from model. model_path shoule be like /path/to/model.pkl'''
#         '''resume from model. model_path shoule be like /path/to/model.pth in our method'''
# 		# self.model.load_state_dict(torch.load(model_path), strict=False)
# 		# print(('successfully resume model from {}'.format(model_path)))
# 		state_dict = torch.load(model_path)
# 		model_dict = self.model.state_dict()
# 		new_state_dict = OrderedDict()
# 		matched_layers, discarded_layers = [], []
# 		for k, v in state_dict.items():
# 			if k.startswith('module.'):
# 				k = k[7:]  # discard module.
# 			if k in model_dict and model_dict[k].size() == v.size():
# 				new_state_dict[k] = v
# 				matched_layers.append(k)
# 			else:
# 				discarded_layers.append(k)
# 		model_dict.update(new_state_dict)
# 		self.model.load_state_dict(model_dict)
# 		if len(discarded_layers) > 0:
# 			print('discarded layers: {}'.format(discarded_layers))

# #######################################################################
# # Evaluate
# parser = argparse.ArgumentParser(description='Demo')
# # which query you want to test. You may select a number in the range of 0 ~ 3367 in market.
# parser.add_argument('--query_index', default=800, type=int, help='test_image_index')
# # dataset url
# # parser.add_argument('--test_dir', default='E:/PytchProject/Person_reID_baseline_pytorch-master'
# #                                           '/Person_reID_baseline_pytorch-master/Market/pytorch', type=str,
# #                     help='./test_data')
# parser.add_argument('--test_dir', default='E:/KUST/NewKU/Trans_a_t2s/data/market1501'
#                     , type=str, help='./test_data')
#
# opts = parser.parse_args()
#
# data_dir = opts.test_dir
# # gallery = bounding_box_test
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x)) for x in ['bounding_box_test', 'query']}
#
#
# #####################################################################

#####################################################################
# Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


######################################################################

######################################################################
def do_vis(cfg,
           model,
           class_net,
           class_net2,
           center_criterion,
           train_loader,
           val_loader,
           optimizer,
           optimizer_center,
           scheduler,
           loss_fn,
           num_query, local_rank, class_optimizer, class2_optimizer, class_scheduler, class2_scheduler,
           target_train_loader, ide_creiteron, loss_classifier):
    model = model.eval()

    # [iters,train_loader]->train_loader

    model_path = 'E:/KUST/NewKU/t2sBack/Trans_two_t2s_101321/logs/2_classifers/bestResult_72.pth'
    if model.load_state_dict(torch.load(model_path), strict=False):
        print(('successfully resume model from {}'.format(model_path)))

    state_dict = torch.load(model_path)
    model_dict = model.state_dict()

    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]  # discard module.
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    if len(discarded_layers) > 0:
        print('discarded layers: {}'.format(discarded_layers))

    # eval model
    model = model.eval()
    query_features_meter, query_pids_meter, query_cids_meter = meter.CatMeter(), meter.CatMeter(), meter.CatMeter()
    gallery_features_meter, gallery_pids_meter, gallery_cids_meter = meter.CatMeter(), meter.CatMeter(), meter.CatMeter()

    # init dataset
    if cfg.DATASETS.TARGET == 'market':
        _datasets = [loaders.market_query_samples, loaders.market_gallery_samples]
        _loaders = [loaders.market_query_loader, loaders.market_gallery_loader]
    elif cfg.DATASETS.TARGET == 'dukemtmc':
        _datasets = [loaders.duke_query_samples, loaders.duke_gallery_samples]
        _loaders = [loaders.duke_query_loader, loaders.duke_gallery_loader]

    # result = scipy.io.loadmat('pytorch_result.mat')
    # query_feature = torch.FloatTensor(result['query_f'])
    # query_cam = result['query_cam'][0]
    # query_label = result['query_label'][0]
    # gallery_feature = torch.FloatTensor(result['gallery_f'])
    # gallery_cam = result['gallery_cam'][0]
    # gallery_label = result['gallery_label'][0]
    #
    # multi = os.path.isfile('multi_query.mat')
    #
    # if multi:
    #     m_result = scipy.io.loadmat('multi_query.mat')
    #     mquery_feature = torch.FloatTensor(m_result['mquery_f'])
    #     mquery_cam = m_result['mquery_cam'][0]
    #     mquery_label = m_result['mquery_label'][0]
    #     mquery_feature = mquery_feature.cuda()
    #
    # query_feature = query_feature.cuda()
    # gallery_feature = gallery_feature.cuda()

#######################################################################
