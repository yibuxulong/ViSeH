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

def set_optimizer(model, opt):
    optimizer = optim.Adam(model.parameters(), weight_decay=opt.weight_decay, lr=opt.lr)
    return optimizer

def lr_scheduler(epoch, optimizer, lr_decay_iter, decay_rate):
    if not (epoch % lr_decay_iter):
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = optimizer.param_groups[i]['lr'] * decay_rate

#---------------------------------------------------- functions in training stage 1 ----------------------------------------------------------------------
def train_fvsa(epoch, data_loader, model, optimizer, opt):
    acc1_v = AverageMeter('Acc@1', ':6.2f')
    acc5_v = AverageMeter('Acc@5', ':6.2f')
    acc1_s = AverageMeter('Acc@1', ':6.2f')
    acc5_s = AverageMeter('Acc@5', ':6.2f')
    losses = AverageMeter('Loss', ':.4e')
    
    hit_cls = np.zeros(opt.num_cls)

    model.train()
    total_time = time.time()
    
    theta_scale_upperbound = opt.tsu
    theta_scale_lowerbound = opt.tsl
    weight_upper_loss = opt.wul
    weight_lower_loss = opt.wll

    weight_diverse_loss = opt.wdl
    weight_anti_outlier_loss_shift = opt.waols
    weight_anti_outlier_loss_bounds = opt.waolb
    weight_rotate_loss = opt.wrl
    for batch_idx, (data, label) in enumerate(data_loader):
        start_time = time.time()           
        [img, words] = data
        # prediction and loss
        batch_size_cur = img.size(0)

        img = img.cuda()
        label = label.cuda()
        #perform prediction
        
        output_s, _, hidden_vectors, theta_list = model(img, opt.dataset_max_seq)
        
        loss_cls_s, batch_vector, label_word, seq_len_list, theta = loss.loss_for_gru_prediction(output_s, words, theta_list)
        loss_cls_s = loss_cls_s * opt.w_semantic
        
        scale_upperbound_loss = loss.get_scale_upperbound_loss([theta[:,0,0], theta[:,1,1]], theta_scale_upperbound, weight_upper_loss)
        scale_lowerbound_loss = loss.get_scale_lowerbound_loss([theta[:,0,0], theta[:,1,1]], theta_scale_lowerbound, weight_lower_loss)
        
        anti_outlier_loss_shift = loss.get_anti_outlier_loss([theta[:,0,2], theta[:,1,2]], weight_anti_outlier_loss_bounds)
        anti_outlier_loss_bounds = loss.get_anti_outlier_loss([theta[:,0,0], theta[:,1,1]], weight_anti_outlier_loss_shift)

        diverse_loss = loss.get_diverse_loss([theta[:,0,2], theta[:,1,2]], seq_len_list, weight_diverse_loss)
        rotate_loss = torch.mean((torch.abs(theta[:,0,1]) + torch.abs(theta[:,1,0])) * 0.5) * weight_rotate_loss
        loss_stn = scale_upperbound_loss + scale_lowerbound_loss + diverse_loss + anti_outlier_loss_shift + anti_outlier_loss_bounds + rotate_loss
        final_loss = loss_cls_s + loss_stn
            
        # optimization
        losses.update(final_loss.item(), batch_size_cur)
        
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()
        
        # upddate log  
        [acc_top1, acc_top5],_ = accuracy_hit(batch_vector, label_word, opt.num_words, topk=(1, 5))
        acc1_s.update(acc_top1[0], batch_size_cur)
        acc5_s.update(acc_top5[0], batch_size_cur)
        
        log_out = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {data_time:.3f}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'loss_s:{loss_s:.4f}, loss_other:{loss_stn:.4f}\t'
                   'Acc@1s {acc1_s.val:.4f} ({acc1_s.avg:.4f})\t'
                  'Acc@5s {acc5_s.val:.4f} ({acc5_s.avg:.4f})\t'.format(
            epoch, batch_idx, len(data_loader), data_time=round((time.time() - total_time), 4), loss=losses, loss_s=loss_cls_s.item(), loss_stn=loss_stn.item(), acc1_s=acc1_s, acc5_s=acc5_s, lr=optimizer.param_groups[-1]['lr']))
        print(log_out)

def collect_feature_word_label(predicts_t, hidden_vectors, words, labels, theta_list):
    # get valid entries
    vector_list = []
    predict_list = []
    label_word_list = []
    label_cls_list = []
    seq_len_list = []
    new_theta_list = []
    theta_list = theta_list.transpose(0, 1)
    words = words.cpu().numpy()
    labels = labels.cpu().numpy()
    
    for i in range(len(words)):# for each batch item
        # get valid seq
        valid_indexes = np.where(words[i]>0)[0]  
        seq_len = len(valid_indexes)
        seq_len_list.append(seq_len)
        # append vectors and labels
        predict_list.append(predicts_t[i,:seq_len])
        vector_list.append(hidden_vectors[i,:seq_len])
        new_theta_list.append(theta_list[i,:seq_len])
        label_word_list.append(valid_indexes)
        label_cls_list.append(labels[i].repeat(seq_len))
    
    batch_vector = torch.cat(vector_list, dim = 0)
    batch_predict = torch.cat(predict_list, dim = 0)
    label_word_list = torch.from_numpy(np.concatenate(label_word_list))
    label_word_list = label_word_list.cuda()
    label_cls_list = torch.from_numpy(np.concatenate(label_cls_list))
    label_cls_list = label_cls_list.cuda()
    return batch_predict, batch_vector, label_word_list, label_cls_list, torch.cat(new_theta_list, dim = 0)

def generate_feature_fvsa(data_loader, model, mode, opt):
    acc1_s = AverageMeter('Acc@1', ':6.2f')
    acc5_s = AverageMeter('Acc@5', ':6.2f')
    losses = AverageMeter('Loss', ':.4e')
    
    hit_cls = np.zeros(opt.num_cls)
    
    model.eval()
    total_time = time.time() 
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(data_loader):
            start_time = time.time()           
            [img, words] = data
            # prediction and loss
            batch_size_cur = img.size(0)
            img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            #perform prediction
            output_s, _,hidden_vectors, theta_list = model(img, opt.dataset_max_seq)
            
            predicts_cur, hidden_vectors_cur, label_word, label_cls, theta = collect_feature_word_label(output_s, hidden_vectors.transpose(0,1), words, label, theta_list)
            # upddate log
            [acc_top1, acc_top5],_ = accuracy_hit(predicts_cur, label_word, opt.num_words, topk=(1, 5))
            acc1_s.update(acc_top1[0], batch_size_cur)
            acc5_s.update(acc_top5[0], batch_size_cur)
            
            if batch_idx==0:
                hidden_vectors_all = np.zeros((0, hidden_vectors_cur.shape[1]))
                label_word_all = np.zeros((0))
                label_cls_all = np.zeros((0))
                theta_all = np.zeros((0,2,3))
                
            hidden_vectors_all = np.concatenate((hidden_vectors_all, hidden_vectors_cur.cpu().numpy()),0)
            label_word_all = np.concatenate((label_word_all, label_word.cpu().numpy()),0)
            label_cls_all = np.concatenate((label_cls_all, label_cls.cpu().numpy()),0)
            theta_all = np.concatenate((theta_all, theta.cpu().numpy()),0)
            
    log_out = ('Acc@1 {acc1_s.avg:.4f}\t Acc@5 {acc5_s.avg:.4f}\t'.format(acc1_s=acc1_s, acc5_s=acc5_s))
    print(log_out)
    # save
    # np.save(opt.result_path+'/valid_hidden_vectors.npy', hidden_vectors_all)
    np.save(opt.result_path+'/hidden_vector_wordIDs.npy', label_word_all)
    np.save(opt.result_path+'/hidden_vector_classIDs.npy', label_cls_all)
    np.save(opt.result_path+'/theta.pt', theta_all)
    
    #perform min-max normalization to hidden vectors for ART clustering
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler((-1,1))
    hidden_vector_scaled = min_max_scaler.fit_transform(hidden_vectors_all)

    feature_max = np.max(hidden_vector_scaled,0)
    feature_min = np.min(hidden_vector_scaled,0)
    
    #tensor_normalizer = {'feature_max': feature_max, 'feature_min': feature_min}

    np.savez(opt.result_path+'/tensor_normalizer.npz', feature_max=feature_max, feature_min=feature_min)

    np.save(opt.result_path+'/scaled_hidden_vectors.npy', hidden_vector_scaled)
        
#---------------------------------------------------- functions in training stage 2 ----------------------------------------------------------------------

def get_decision_of_vshc(data_loader, model, mode, opt):
    acc1_v = AverageMeter('Acc@1', ':6.2f')
    acc5_v = AverageMeter('Acc@5', ':6.2f')
    
    hit_cls = np.zeros(opt.num_cls)
    model.eval()
    total_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(data_loader):
            start_time = time.time()           
            [img, words] = data
            # prediction and loss
            batch_size_cur = img.size(0)
            img = img.cuda()
            label = label.cuda()
            #perform prediction
            model_predict_words_predicts, feature_v, decision_words, decision_classes_topk = model(img, opt.dataset_max_seq)
            output_v = decision_classes_topk[:, 0]
            # upddate log  
            [acc_top1, acc_top5],_ = accuracy_hit(F.one_hot(output_v, opt.num_cls), label, opt.num_cls, topk=(1, 5))
            acc1_v.update(acc_top1[0], batch_size_cur)
            acc5_v.update(acc_top5[0], batch_size_cur)
            
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

            log_out = ('{} [{}/{}], Time {data_time:.3f}\t'
                       'Acc@1v {acc1_v.val:.4f} ({acc1_v.avg:.4f})\t'
                      'Acc@5v {acc5_v.val:.4f} ({acc5_v.avg:.4f})\t'.format(mode, batch_idx, len(data_loader), data_time=round((time.time() - total_time), 4), acc1_v=acc1_v, acc5_v=acc5_v))
            print(log_out)
    
    log_out = ('overall: Acc@1v {acc1_v.val:.4f} ({acc1_v.avg:.4f})\t Acc@5v {acc5_v.val:.4f} ({acc5_v.avg:.4f})\t'.format(
            mode, batch_idx, len(data_loader), data_time=round((time.time() - total_time), 4), acc1_v=acc1_v, acc5_v=acc5_v))
    # save result
    torch.save(model_predict_words_predicts_all, opt.result_path+'/model_predict_words_predicts_{}.pt'.format(mode))
    torch.save(feature_v_all, opt.result_path+'/feature_v_all_{}.pt'.format(mode))
    torch.save(decision_words_all, opt.result_path+'/decision_words_all_{}.pt'.format(mode))
    torch.save(decision_classes_topk_all, opt.result_path+'/decision_classes_topk_all_{}.pt'.format(mode))
    torch.save(label_all, opt.result_path+'/label_all_{}.pt'.format(mode)) 
    
    
def train_mmgf(epoch, data_loader, model, optimizer, opt):
    acc1_v = AverageMeter('Acc@1', ':6.2f')
    acc5_v = AverageMeter('Acc@5', ':6.2f')
    losses = AverageMeter('Loss', ':.4e')
    
    hit_cls = np.zeros(opt.num_cls)
    model.train()
    total_time = time.time()
    for batch_idx, (data, label) in enumerate(data_loader):
        [feature_v, word_predicts, decision_classes_topk, decision_words] = data        
        start_time = time.time()           
        # prediction and loss
        batch_size_cur = label.shape[0]
        
        label = label.long().cuda()
        word_predicts = word_predicts.cuda()
        decision_classes_topk = decision_classes_topk.cuda()
        decision_words = decision_words.cuda()
        feature_v = feature_v.cuda()
        
        _, output_v = model(word_predicts, feature_v, decision_words)
            
        loss_cls_v = loss.loss_for_img_classification(output_v, label)
        final_loss = loss_cls_v
            
        # optimization
        losses.update(final_loss.item(), batch_size_cur)
        
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()
        
        # upddate log  
        [acc_top1, acc_top5],_ = accuracy_hit(output_v, label, opt.num_cls, topk=(1, 5))
        acc1_v.update(acc_top1[0], batch_size_cur)
        acc5_v.update(acc_top5[0], batch_size_cur)
        
        optimizer_cur = optimizer
        log_out = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {data_time:.3f}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'loss_v:{loss_v:.4f}\t'
                   'Acc@1v {acc1_v.val:.4f} ({acc1_v.avg:.4f})\t'
                  'Acc@5v {acc5_v.val:.4f} ({acc5_v.avg:.4f})\t'.format(
            epoch, batch_idx, len(data_loader), data_time=round((time.time() - total_time), 4), loss=losses, loss_v=loss_cls_v.item(), acc1_v=acc1_v, acc5_v=acc5_v, lr=optimizer_cur.param_groups[-1]['lr']))
        print(log_out)
        
def generate_feature_mmgf(data_loader, model):
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(data_loader):
            [feature_v, word_predicts, decision_classes_topk, decision_words] = data                          
            label = label.long().cuda()
            word_predicts = word_predicts.cuda()
            decision_classes_topk = decision_classes_topk.cuda()
            decision_words = decision_words.cuda()
            feature_v = feature_v.cuda()

            feature_mmgf, _ = model(word_predicts, feature_v, decision_words)
            if batch_idx==0:
                feature_mmgf_all = torch.zeros((0, feature_mmgf.shape[1]))
                label_all = torch.zeros(0)
            feature_mmgf_all = torch.cat((feature_mmgf_all, feature_mmgf.cpu()), 0)
            label_all = torch.cat((label_all, label.cpu()), 0)
    print('MMGF features generated.')
    return feature_mmgf_all.detach().cpu(), label_all.detach().cpu()

def generate_feature_global(data_loader, model):
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(data_loader):  
            [img, words] = data
            img = img.cuda()
            label = label.cuda()
            #perform prediction
            feature_global, _ = model.forward_generate(img)
            
            if batch_idx==0:
                feature_global_all = torch.zeros((0, feature_global.shape[1]))
                label_all = torch.zeros(0)
            feature_global_all = torch.cat((feature_global_all, feature_global.cpu()), 0)
            label_all = torch.cat((label_all, label.cpu()), 0)
    print('Global features generated.')
    return feature_global_all.detach().cpu(), label_all.detach().cpu()

        
def train_global(epoch, data_loader, model, optimizer, opt):
    
    # vireo measurement
    acc1_v = AverageMeter('Acc@1', ':6.2f')
    acc5_v = AverageMeter('Acc@5', ':6.2f')
    acc10_v = AverageMeter('Acc@5', ':6.2f')
    losses = AverageMeter('Loss', ':.4e')
    
    hit_cls = np.zeros(opt.num_cls)

    model.train()
    total_time = time.time()
    for batch_idx, (data, label) in enumerate(data_loader):
        start_time = time.time()           
        [img, words] = data
        # prediction and loss
        batch_size_cur = img.size(0)
        img = img.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        #perform prediction
        output = model(img)
        
        criterion = nn.CrossEntropyLoss()
        loss_cls_v = criterion(output, label)

        final_loss = loss_cls_v
            
        # optimization
        losses.update(final_loss.item(), batch_size_cur)
        
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()
        
        # upddate log
        [acc_top1, acc_top5, acc_top10],_ = accuracy_hit(output, label, opt.num_cls, topk=(1, 5, 10))
        acc1_v.update(acc_top1[0], batch_size_cur)
        acc5_v.update(acc_top5[0], batch_size_cur)
        acc10_v.update(acc_top10[0], batch_size_cur)
            
        optimizer_cur = optimizer
        log_out = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {data_time:.3f}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {acc1_v.val:.4f} ({acc1_v.avg:.4f})\t'
                  'Acc@5 {acc5_v.val:.4f} ({acc5_v.avg:.4f})\t'
                  'Acc@10 {acc10_v.val:.4f} ({acc10_v.avg:.4f})\t'.format(
            epoch, batch_idx, len(data_loader), data_time=round((time.time() - total_time), 4), loss=losses, acc1_v=acc1_v, acc5_v=acc5_v, acc10_v=acc10_v, lr=optimizer_cur.param_groups[-1]['lr']))
        print(log_out)
        
def train_fusion(epoch, data_loader, model, optimizer, opt):
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    losses = AverageMeter('Loss', ':.4e')
    model.train()
    total_time = time.time()
    for batch_idx, (data, label) in enumerate(data_loader):
        [feature_global, feature_mmgf] = data        
        start_time = time.time()           
        # prediction and loss
        batch_size_cur = label.shape[0]
        label = label.long().cuda()
        feature_global = feature_global.cuda()
        feature_mmgf = feature_mmgf.cuda()
        
        _, pre_fusion = model(feature_global, feature_mmgf)
        # loss
        criterion_cls = nn.CrossEntropyLoss()
        loss_fusion = criterion_cls(pre_fusion, label)
        losses.update(loss_fusion.item(), batch_size_cur)
        # update model
        optimizer.zero_grad()
        loss_fusion.backward()
        optimizer.step()

        # metrix and output
        acc1, acc5 = accuracy(pre_fusion, label, topk=(1, 5))
        top1.update(acc1[0], batch_size_cur)
        top5.update(acc5[0], batch_size_cur)

        log_out = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                   'Time {data_time:.3f}\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, batch_idx, len(data_loader),
                                                                  data_time=round((time.time() - total_time), 4),
                                                                  loss=losses, top1=top1, top5=top5,
                                                                  lr=optimizer.param_groups[-1]['lr']))
        print(log_out)