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
import torch.distributed as dist
from torch.nn import functional as F
from torchvision import transforms
import build_dataset
from torch import optim
from utils import *
import itertools
from torchvision.utils import save_image
import train_functions
import opts


#------Processing------
# Step 1 Getting raw and knowledge-based predictions of words
# Step 2: Training CAGL model for fusing outputs of FVSA and VSRF
# Step 3: Training the global model (visual backbone)
# Step 4: Later fusion of cross-modal features (from CAGL module) and glbal visual features (for global model).
#----------------------

#------Settings------
opt = opts.opt_algorithm()
opt.path_root = '/data_food101/' # path to root folder
opt.path_img = '/data_images_food101/'# path to image folder
opt.path_data = opt.path_root # path to data folder
opt.path_class_name = opt.path_data + 'classes.txt'# path to the list of names for classes
opt.num_cls = 101 # number of classes in the dataset
opt.dataset_max_seq = 25 # max number of word for a sample in train data
opt.num_words = 446 # number of ingredients in the dataset
opt.size_img = [384, 384]

CUDA = 1
SEED = 1
measure_best = 0 # best measurement
epoch_best = 0
torch.manual_seed(SEED)
kwargs = {'num_workers': 0, 'pin_memory': True} if CUDA else {}
if CUDA:
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)
    
# log and model paths
result_path = opt.result_path
prepare_intermediate_folders(result_path)
model_save_path = 'model_save/{}/'.format(opt.net_v)
prepare_intermediate_folders(model_save_path)


EPOCHS = opt.lr_decay * 3 + 1
#--------------------

#--------------------------------------------Step 1 Getting raw and knowledge-based predictions of words-------------------------------------------------
# dataset for feature & decision generation
transform_img_train = transforms.Compose([
    transforms.Resize([384, 384]),
    transforms.ToTensor(),])

train_dataset = build_dataset.dataset_for_classification(opt, 'train', transform_img_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=False, **kwargs)

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
opt.method = 'vsrf'
model_vsrf = build_model.build(CUDA,opt)

# feature & decision generation
train_functions.get_decision_of_vsrf(train_loader, model_vsrf, 'train', opt)
del model_vsrf, train_dataset, train_loader
#------------------------------------Step 2: Training CAGL module for fusing outputs of FVSA and VSRF----------------------------------------------------
# dataset
label_train = torch.load(opt.result_path+'/label_all_train.pt')
word_predicts_train = torch.load(opt.result_path+'/model_predict_words_predicts_train.pt')
decision_classes_topk_train = torch.load(opt.result_path+'/decision_classes_topk_all_train.pt')
decision_words_train = torch.load(opt.result_path+'/decision_words_all_train.pt')
feature_v_train = torch.load(opt.result_path+'/feature_v_all_train.pt')

import build_dataset
train_dataset_cagl = build_dataset.dataset_for_cagl(label_train, feature_v_train, word_predicts_train, decision_classes_topk_train, decision_words_train)
train_loader_cagl = torch.utils.data.DataLoader(train_dataset_cagl, batch_size=opt.batch_size, shuffle=True, **kwargs)

# CAGL model
opt.method='cagl'
model_cagl = build_model.build(CUDA,opt)
optimizer_cagl = train_functions.set_optimizer(model_cagl,opt)

# raining

for epoch in range(1, EPOCHS + 1):
    train_functions.lr_scheduler(epoch, optimizer_cagl, opt.lr_decay, opt.lrd_rate)
    train_functions.train_cagl(epoch, train_loader_cagl, model_cagl, optimizer_cagl, opt)

torch.save(model_cagl.state_dict(), model_save_path + 'model_cagl.pt')

#--------------------------------------------------------Step 3: Training global model----------------------------------------------------------------
# dataset
train_dataset_global = build_dataset.dataset_for_classification(opt, 'train', transform_img_train)
train_loader_global = torch.utils.data.DataLoader(train_dataset_global, batch_size=opt.batch_size, shuffle=True, **kwargs)

# global model
opt.method='global'
model_global = build_model.build(CUDA,opt)
optimizer_global = train_functions.set_optimizer(model_global,opt)
opt.lr=5e-5

for epoch in range(1, EPOCHS + 1):
    train_functions.lr_scheduler(epoch, optimizer_global, opt.lr_decay, opt.lrd_rate)
    train_functions.train_global(epoch, train_loader_global, model_global, optimizer_global, opt)
    
torch.save(model_global.state_dict(), model_save_path + 'model_global.pt')

#---------------------------------------------------------------Step 4: Later fusion----------------------------------------------------------------------
# dataset
train_loader_cagl = torch.utils.data.DataLoader(train_dataset_cagl, batch_size=100, shuffle=False, **kwargs)
train_loader_global = torch.utils.data.DataLoader(train_dataset_global, batch_size=100, shuffle=False, **kwargs)
feature_cagl, label_cagl = train_functions.generate_feature_cagl(train_loader_cagl, model_cagl)
feature_global, label_global = train_functions.generate_feature_global(train_loader_global, model_global)

train_dataset_fusion = build_dataset.dataset_for_fusion(feature_global, feature_cagl, label_global)
train_loader_fusion = torch.utils.data.DataLoader(train_dataset_fusion, batch_size=opt.batch_size, shuffle=True, **kwargs)

# model
opt.method='fusion'
model_fusion = build_model.build(CUDA,opt)
optimizer_fusion = train_functions.set_optimizer(model_fusion, opt)
lr = 1e-3

for epoch in range(1, EPOCHS + 1):
    train_functions.lr_scheduler(epoch, optimizer_fusion, opt.lr_decay, opt.lrd_rate)
    train_functions.train_fusion(epoch, train_loader_fusion, model_fusion, optimizer_fusion, opt)


torch.save(model_fusion.state_dict(), model_save_path + 'model_fusion.pt')