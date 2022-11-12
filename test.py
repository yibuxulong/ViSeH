# -*- coding: utf-8 -*-
import io
import os
import os.path
import time
import argparse
import torch
import loss
import torch.utils.data
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
import build_dataset
from utils import *
import test_functions
import opts


#------Processing------
# Load 1: Load raw and knowledge-based predictions of words
# Load 2: Load CAGL model
# Load 3: Load Global model
# Predictions of Global model, CAGL model, later fusion, and decision fusion
#----------------------

#------Settings------
opt = opts.opt_algorithm()
opt.path_root = 'data_food101_demo/' # path to root folder
opt.path_img = 'data_food101_demo/images/'# path to image folder
opt.path_data = opt.path_root # path to data folder
opt.path_class_name = opt.path_data + 'classes.txt'# path to the list of names for classes
opt.num_cls = 101 # number of classes in the dataset
opt.dataset_max_seq = 25 # max number of word for a sample in train data
opt.num_words = 446 # number of ingredients in the dataset
opt.size_img = [384, 384]

CUDA = 1
SEED = 1

torch.manual_seed(SEED)
kwargs = {'num_workers': 0, 'pin_memory': True} if CUDA else {}
if CUDA:
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)
    
# log and model paths
result_path = opt.result_path
if not os.path.exists(result_path):
    os.makedirs(result_path)
model_save_path = 'model_save/{}/'.format(opt.net_v)

#--------------------

#--------------------------------------------Load 1: Load raw and knowledge-based predictions of words-------------------------------------------------
# dataset for feature & decision generation
transform_img_test = transforms.Compose([
    transforms.Resize([384, 384]),
    transforms.ToTensor(),])

test_dataset = build_dataset.dataset_for_classification(opt, 'test', transform_img_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, **kwargs)

# model for feature & decision generation
opt.path_fvsa = model_save_path + 'model_fvsa.pt'
select_epoch = opt.art_epoch
path_hierarchy_root = opt.hierarchy_path + str(opt.art_rho_0) + '~' + str(opt.art_beta)+'/'+str(opt.art_P_T)+'/'
path_feature_root = opt.hierarchy_path
hierarchy_net_path = path_hierarchy_root + 'cleaned_networks_{}.npz'.format(select_epoch)

# cluster hierarchy
networks = np.load(hierarchy_net_path)
opt.cluster_weights = torch.from_numpy(networks['W_new']).cuda()
opt.cluster_label_indicators = torch.from_numpy(networks['cluster_label_indicators_new']).cuda()

tensor_normalizer = np.load(path_feature_root + 'tensor_normalizer.npz')
opt.valid_cluster_class = torch.from_numpy(np.load(path_hierarchy_root + 'network_{}_valid_cluster_class.npy'.format(select_epoch))).cuda()
opt.feature_max = torch.from_numpy(tensor_normalizer['feature_max']).cuda()
opt.feature_min = torch.from_numpy(tensor_normalizer['feature_min']).cuda()

# model define
import build_model
# modify
# opt.method = 'vshc'
opt.method = 'vsrf'
# model_vshc = build_model.build(CUDA,opt)
model_vsrf = build_model.build(CUDA,opt)

# feature & decision generation
# modify
# test_functions.get_decision_of_vshc(test_loader, model_vshc, 'test', opt)
test_functions.get_decision_of_vsrf(test_loader, model_vsrf, 'test', opt)
# del model_vshc, test_dataset, test_loader
del model_vsrf, test_dataset, test_loader
#----------------------------------------------------------Load 2: Load CAGL module---------------------------------------------------------------------
# dataset
label_test = torch.load(opt.result_path+'/label_all_test.pt')
word_predicts_test = torch.load(opt.result_path+'/model_predict_words_predicts_test.pt')
decision_classes_topk_test = torch.load(opt.result_path+'/decision_classes_topk_all_test.pt')
decision_words_test = torch.load(opt.result_path+'/decision_words_all_test.pt')
feature_v_test = torch.load(opt.result_path+'/feature_v_all_test.pt')

import build_dataset
# modify
test_dataset_cagl = build_dataset.dataset_for_cagl(label_test, feature_v_test, word_predicts_test, decision_classes_topk_test, decision_words_test)
test_loader_cagl = torch.utils.data.DataLoader(test_dataset_cagl, batch_size=100, shuffle=False, **kwargs)

# load CAGL model
opt.method='cagl'
model_cagl = build_model.build(CUDA,opt)
# modify model name
model_cagl = build_model.get_updateModel(model_cagl, model_save_path + 'model_cagl.pt')

#--------------------------------------------------------Load 3: Load Global module----------------------------------------------------------------
# dataset
test_dataset_global = build_dataset.dataset_for_classification(opt, 'test', transform_img_test)
test_loader_global = torch.utils.data.DataLoader(test_dataset_global, batch_size=100, shuffle=False, **kwargs)

# global model
opt.method='global'
model_global = build_model.build(CUDA,opt)
model_global = build_model.get_updateModel(model_global, model_save_path + 'model_global.pt')

#--------------------------------------Predictions of Global model, CAGL model, later fusion, and decision fusion ----------------------------------------
# dataset
test_loader_cagl = torch.utils.data.DataLoader(test_dataset_cagl, batch_size=100, shuffle=False, **kwargs)
test_loader_global = torch.utils.data.DataLoader(test_dataset_global, batch_size=100, shuffle=False, **kwargs)
feature_global, output_global, label_global = test_functions.generate_feature_global(test_loader_global, model_global, opt)
feature_cagl, output_cagl, label_cagl = test_functions.generate_feature_cagl(test_loader_cagl, model_cagl, opt)

test_dataset_fusion = build_dataset.dataset_for_fusion(feature_global, feature_cagl, label_global)
test_loader_fusion = torch.utils.data.DataLoader(test_dataset_fusion, batch_size=100, shuffle=False, **kwargs)

# model
opt.method='fusion'
model_fusion = build_model.build(CUDA,opt)
model_fusion = build_model.get_updateModel(model_fusion, model_save_path + 'model_fusion.pt')
output_fusion = test_functions.test_fusion(test_loader_fusion, model_fusion, opt)

beta_fusion = opt.beta_fusion
refined_decision = beta_fusion*nn.Softmax(1)(F.one_hot(decision_classes_topk_test[:,0].long(), opt.num_cls).float())+(1.-beta_fusion)*nn.Softmax(1)(output_fusion)


# Show predictions from different models
top_k_pre = 5
label = label_global
pre_cagl = torch.topk(output_cagl, top_k_pre)[1]
pre_global = torch.topk(output_global, top_k_pre)[1]
# modify
# pre_vshc = decision_classes_topk_test[:,:top_k_pre]
pre_vsrf = decision_classes_topk_test[:,:top_k_pre]
pre_refine = torch.topk(refined_decision, top_k_pre)[1]

for i in range(label.shape[0]):
    print('Sample {}, Ground Truth label: {}'.format(i, label[i].long().item()))
    for k in range(top_k_pre):
        if k==0:
            print('Top-{} Class prediction: Global model: [{}], VSRF: [{}], CAGL: [{}] | final prediction: [{}]'.format(k+1, pre_global[i][k], pre_vsrf[i][k].long(), pre_cagl[i][k], pre_refine[i][k]))
        else:
            print('Top-{} Class prediction: Global model: [{}], VSRF: [{}], CAGL: [{}]'.format(k+1, pre_global[i][k], pre_vsrf[i][k].long(), pre_cagl[i][k]))