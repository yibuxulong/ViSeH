# -*- coding: utf-8 -*-
import io
import os
import os.path
import time
import torch
import loss
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from utils import *

def get_decision_of_vsrf(data_loader, model, mode, opt):
    model.eval()
    total_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(data_loader): 
            [img, words] = data
            # prediction and loss
            batch_size_cur = img.size(0)
            img = img.cuda()
            label = label.cuda()
            #perform prediction
            model_predict_words_predicts, feature_v, decision_words, decision_classes_topk = model(img, opt.dataset_max_seq)
            output_v = decision_classes_topk[:, 0]
            # upddate log  
            if batch_idx==0:
                model_predict_words_predicts_all = torch.zeros((0, opt.dataset_max_seq, opt.num_words))
                feature_v_all = torch.zeros((0, feature_v.shape[1]))
                decision_words_all = torch.zeros((0, opt.top_seq))
                decision_classes_topk_all = torch.zeros((0, opt.topk))
                label_all = torch.zeros(0)

            model_predict_words_predicts_all = torch.cat((model_predict_words_predicts_all, model_predict_words_predicts.cpu()),0)
            feature_v_all = torch.cat((feature_v_all, feature_v.cpu()),0)
            decision_words_all = torch.cat((decision_words_all, decision_words.cpu()),0)
            decision_classes_topk_all = torch.cat((decision_classes_topk_all, decision_classes_topk.cpu()),0)
            label_all = torch.cat((label_all, label.cpu()))
            
    # save result
    torch.save(model_predict_words_predicts_all, opt.result_path+'/model_predict_words_predicts_{}.pt'.format(mode))
    torch.save(feature_v_all, opt.result_path+'/feature_v_all_{}.pt'.format(mode))
    torch.save(decision_words_all, opt.result_path+'/decision_words_all_{}.pt'.format(mode))
    torch.save(decision_classes_topk_all, opt.result_path+'/decision_classes_topk_all_{}.pt'.format(mode))
    torch.save(label_all, opt.result_path+'/label_all_{}.pt'.format(mode)) 
    
def test_fusion(data_loader, model, opt):
    model.eval()
    total_time = time.time()
    for batch_idx, (data, label) in enumerate(data_loader):        
        batch_size_cur = label.shape[0]
        [feature_global, feature_cagl] = data                  
        
        label = label.long().cuda()
        feature_global = feature_global.cuda()
        feature_cagl = feature_cagl.cuda()
        
        _, pre_fusion = model(feature_global, feature_cagl)
        
        if batch_idx==0:
            pre_fusion_all = torch.zeros((0, opt.num_cls))
        pre_fusion_all = torch.cat((pre_fusion_all, pre_fusion.cpu()), 0)
    
    return pre_fusion_all
    
def generate_feature_cagl(data_loader, model, opt):
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(data_loader):
            
            batch_size_cur = label.shape[0]
            [feature_v, word_predicts, decision_classes_topk, decision_words] = data  
            
            label = label.long().cuda()
            word_predicts = word_predicts.cuda()
            decision_classes_topk = decision_classes_topk.cuda()
            decision_words = decision_words.cuda()
            feature_v = feature_v.cuda()

            feature_cagl, output_cagl = model(word_predicts, feature_v, decision_words)
            
            if batch_idx==0:
                feature_cagl_all = torch.zeros((0, feature_cagl.shape[1]))
                output_cagl_all = torch.zeros((0, opt.num_cls))
                label_all = torch.zeros(0)
            feature_cagl_all = torch.cat((feature_cagl_all, feature_cagl.cpu()), 0)
            output_cagl_all = torch.cat((output_cagl_all, output_cagl.cpu()), 0)
            label_all = torch.cat((label_all, label.cpu()), 0)

    print('CAGL features generated.')

    return feature_cagl_all.detach().cpu(), output_cagl_all.detach().cpu(), label_all.detach().cpu()

def generate_feature_global(data_loader, model, opt):    
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(data_loader):
            batch_size_cur = label.shape[0]
            [img, words] = data
            img = img.cuda()
            label = label.cuda()
            
            feature_global, output_global = model.forward_generate(img)
            
            if batch_idx==0:
                feature_global_all = torch.zeros((0, feature_global.shape[1]))
                output_global_all = torch.zeros((0, opt.num_cls))
                label_all = torch.zeros(0)
            feature_global_all = torch.cat((feature_global_all, feature_global.cpu()), 0)
            output_global_all = torch.cat((output_global_all, output_global.cpu()), 0)
            label_all = torch.cat((label_all, label.cpu()), 0)
            
    print('Global features generated.')
    
    return feature_global_all.detach().cpu(), output_global_all.detach().cpu(), label_all.detach().cpu()

