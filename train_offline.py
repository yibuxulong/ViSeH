# -*- coding: utf-8 -*-
import io
import os
import os.path
import time
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
import train_functions
#load environmental settings
import opts
opt = opts.opt_algorithm()
#------Processing------
# Step 1: Training FVSA model for paring visual region and semantic tags
# Step 2: Clustering the visual-semantic pairs
# Step 3: Filtering the clusters and creating the Hierarhcy
#----------------------

#------Settings------
opt.dataset = 'food101'
opt.word_net = 'gru'

opt.modality='v+s'
opt.path_root = '/data_food101/' # path to root folder
opt.path_img = '/data_images_food101/'# path to image folder
opt.path_data = opt.path_root # path to data folder
opt.path_class_name = opt.path_data + 'classes.txt'# path to the list of names for classes
opt.num_cls = 101 # number of classes in the dataset
opt.dataset_max_seq = 25 # max number of words for a sample in train data
opt.num_words = 446 # number of words in the dataset
opt.size_img = [384, 384]

# basic
CUDA = 1  # 1 for True; 0 for False
SEED = 1

torch.manual_seed(SEED)
kwargs = {'num_workers': 0, 'pin_memory': True} if CUDA else {}
if CUDA:
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)
# log and model paths
result_path = os.path.join(opt.result_path, para_name(opt))
prepare_intermediate_folders(result_path)
model_save_path = 'model_save/{}/'.format(opt.net_v)
prepare_intermediate_folders(model_save_path)
#train settings
EPOCHS = opt.lr_decay * 3 + 1
    
#----------------------------------------Step 1: Training FVSA model for paring visual region and semantic tags-------------------------------------------
# dataset
transform_img_train = transforms.Compose([
    transforms.Resize([384, 384]),
    transforms.ToTensor(),])

# create dataset
train_dataset = build_dataset.dataset_for_classification(opt, 'train', transform_img_train)

# dataloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, **kwargs)

# Model of FVSA
import build_model
import build_model_hierarchy
opt.method = 'fvsa'
model_fvsa = build_model.build(CUDA,opt)
optimizer_fvsa = train_functions.set_optimizer(model_fvsa,opt)

if opt.net_v=='resnet18':
    dim_feat_v = 512
elif opt.net_v=='resnet50':
    dim_feat_v = 2048
elif opt.net_v=='vit':
    dim_feat_v = 768

# FVSA training
for epoch in range(1, EPOCHS + 1):
    train_functions.lr_scheduler(epoch, optimizer_fvsa, opt.lr_decay, opt.lrd_rate)
    train_functions.train_fvsa(epoch, train_loader, model_fvsa, optimizer_fvsa, opt)

torch.save(model_fvsa.state_dict(), model_save_path + 'model_fvsa.pt')
train_functions.generate_feature_fvsa(train_loader, model_fvsa, 'train', opt)

#-----------------------------------------------------Step 2: Clustering the visual-semantic pairs-------------------------------------------------------
# parameters of VSHC creation
epoch_cluster = opt.art_epoch

cluster_path = opt.result_path + str(opt.art_rho_0) + '~' + str(opt.art_beta)+'/'
cleaned_path = opt.result_path + str(opt.art_rho_0) + '~' + str(opt.art_beta)+'/'+str(opt.art_P_T)+'/'
prepare_intermediate_folders(cluster_path)
prepare_intermediate_folders(cleaned_path)

# ART Dataset
dataset_art = build_dataset.dataset_for_art(opt.num_words, opt.result_path)
data_loader_art = torch.utils.data.DataLoader(dataset_art, batch_size=opt.batch_size, shuffle=True, **kwargs)
# generate clusters
print('Clustering results save to: {}'.format(cluster_path))
print('algorithm settings: rho={} | beta={} | sigma={}'.format(opt.art_rho_0, opt.art_beta, opt.art_sigma))
 
hidden_vector_wordIDs = np.load(opt.result_path+'/hidden_vector_wordIDs.npy')
num_data = hidden_vector_wordIDs.shape[0]
for epoch in range(1, epoch_cluster+1):
    start_time = time.time()
    if epoch == 1:
        networks, data_Assign = build_model_hierarchy.network_init(opt, num_data, opt.num_words, dim_feat_v)
    else:
        networks, data_Assign = build_model_hierarchy.network_clear(opt, networks, data_Assign)

    networks, data_Assign = build_model_hierarchy.clustering_epoch(opt, epoch, networks, data_Assign, data_loader_art)
    build_model_hierarchy.save_results(opt, epoch, cluster_path, start_time, networks, data_Assign)
    
#----------------------------------------------Step 3: Filtering the clusters and creating the Hierarhcy--------------------------------------------------
# filter clusters
net = np.load(cluster_path + 'networks_' + str(epoch) + '.npz')
W, J, L, rhos, cluster_label_indicators = net['W'], net['J'], net['L'], net['rhos'], net['cluster_label_indicators']
data_assign = torch.tensor(np.load(cluster_path + 'data_Assign_' + str(epoch) + '.npy'))
L = torch.tensor(L)

print('Epoach {}: Number of clusters: {} | max_size: {} | cluster_size>1: {} | cluster_size>2: {}'.format(epoch, J, torch.max(L), len(torch.nonzero(L > 0)[:,0].view(-1)), len(torch.nonzero(L > 1)[:,0].view(-1))))

classIDs = np.array(np.load(opt.result_path+'/hidden_vector_classIDs.npy'))  # patch class label
cls_size = torch.sum(F.one_hot(data_assign.long()),dim=0).cpu().numpy().astype(float)

nonempty_cluster_IDs = torch.nonzero(L > 1)[:,0].view(-1)

valid_cluster_IDs = []  # save Clusters ID with dominant categories
valid_cluster_class = []
invalid_cluster_class = []

for clusterid in nonempty_cluster_IDs:
    patch_id = np.where(data_assign==int(clusterid))[0]  # patch ID assigned to cluster j
    class_id = classIDs[patch_id]  # patch - classID

    cluster_class = list(set(class_id))   # classID in cluster

    cluster_class_size = [len(np.where(class_id==k)[0]) for k in cluster_class]   # class size in cluster
    cluster_class_size_max = cluster_class[np.argmax(cluster_class_size)]   # max class in cluster
    lk = len(np.where(class_id == cluster_class_size_max)[0])# max class size in cluster
    # have dominant class
    if lk/cls_size[clusterid] >= opt.art_P_T:
        print('dominant_rate: {}, cluster_ID: {}, dominant_class: {}'.format(lk/cls_size[clusterid], clusterid, cluster_class_size_max))
        valid_cluster_IDs.append(clusterid)
        valid_cluster_class.append(cluster_class_size_max)
# save cluster with dominant class 
print(len(valid_cluster_IDs))
np.save(cleaned_path+'network_'+str(epoch)+'_valid_cluster_IDs.npy', np.array(valid_cluster_IDs))
np.save(cleaned_path+'network_'+str(epoch)+'_valid_cluster_class.npy', np.array(valid_cluster_class))

# clean the network by removing the empty clusters
W_new = W[valid_cluster_IDs]
L_new = L[valid_cluster_IDs]
cluster_label_indicators_new = cluster_label_indicators[valid_cluster_IDs]
J_new = len(valid_cluster_IDs)

np.savez(cleaned_path + 'cleaned_networks_' + str(epoch) + '.npz', W_new=W_new, J_new=J_new, L_new=L_new,
         cluster_label_indicators_new=cluster_label_indicators_new)

data_assign = torch.tensor(np.load(cluster_path + 'data_Assign_'+ str(epoch) + '.npy'))
hidden_vector_scaled = np.load(opt.result_path+'/scaled_hidden_vectors.npy') 
hidden_vector_wordIDs = np.load(opt.result_path+'/hidden_vector_wordIDs.npy')
hidden_vector_classIDs = np.load(opt.result_path+'/hidden_vector_classIDs.npy')

data_valid_IDs = []
valid_cluster_IDs = np.load(cleaned_path+'network_'+str(epoch)+'_valid_cluster_IDs.npy')
for i in range(J):
    if i in valid_cluster_IDs:
        data_valid_id = np.where(data_assign == i)[0]
        data_valid_IDs.extend(list(data_valid_id))
