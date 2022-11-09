import numpy as np
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

criterion_img = nn.CrossEntropyLoss()
softmax = nn.Softmax(dim=1)

def loss_for_img_classification(predicts_v, labels):
    CE_v = criterion_img(predicts_v, labels)
    return CE_v

def loss_for_gru_prediction(predicts_t, words, theta_list):
    # get valid entries
    vector_list = []
    label_list = []
    seq_len_list = []
    new_theta_list = []
    theta_list = theta_list.transpose(0, 1)
    words = words.numpy()
    
    for i in range(len(words)):# for each batch item
        # get valid seq
        valid_indexes = np.where(words[i]>0)[0]  
        seq_len = len(valid_indexes)
        seq_len_list.append(seq_len)
        vector_list.append(predicts_t[i,:seq_len])
        new_theta_list.append(theta_list[i,:seq_len])
        label_list.append(valid_indexes)
    
    batch_vector = torch.cat(vector_list, dim = 0)
    label_list = torch.from_numpy(np.concatenate(label_list))
    label_list = label_list.cuda()
    
    gru_loss = criterion_img(batch_vector, label_list)
    return gru_loss, batch_vector, label_list, np.array(seq_len_list), torch.cat(new_theta_list, dim = 0)


def get_distance_matrix(shift_vector):
    #import ipdb; ipdb.set_trace()
    shift_vector_repeated = shift_vector.unsqueeze(1).repeat(1,len(shift_vector))
    distance_matrix = torch.abs(shift_vector_repeated - shift_vector)
    return distance_matrix
      
def get_diverse_loss(shift, seq_len_list, weight):
    shift_x = shift[0] #(#seq_in_batch,)
    shift_y = shift[1]
    
    diverse_loss_all = 0
    
    #compute the shift diversity for each image
    for i in range(len(seq_len_list)):
        if i == 0:
            start_seq = 0
        else:
            start_seq = seq_len_list[:i].sum()
        
        cur_img_shift_x_list = shift_x[start_seq:(start_seq+seq_len_list[i])]
        cur_img_shift_y_list = shift_y[start_seq:(start_seq+seq_len_list[i])]
        
        distance_matrix_x = get_distance_matrix(cur_img_shift_x_list)
        distance_matrix_y = get_distance_matrix(cur_img_shift_y_list)
        distance_matrix_sum = distance_matrix_x.sum()*0.5 + distance_matrix_y.sum()*0.5
        cur_img_shift_loss = torch.exp(-distance_matrix_sum)
            
        diverse_loss_all += cur_img_shift_loss
        
    return (diverse_loss_all/len(seq_len_list))*weight

def get_distance_vector(point_vector):
    distance_vector = torch.abs(point_vector - point_vector.mean())
    return distance_vector.mean()

def get_anti_outlier_loss(point_vectors, weight):
    loss_all = 0
    for point_vector in point_vectors:
        loss_value = get_distance_vector(point_vector)
        loss_all+= loss_value
    return loss_all * weight

def get_shift_loss(shift, loc_range, weight):
    shift_x = shift[0] #(batch_size, x^t)
    shift_y = shift[1]
    shift_loss = (torch.abs(shift_x) - loc_range)**2 + (torch.abs(shift_y) - loc_range)**2 
    return torch.mean(shift_loss * 0.5)*weight

def get_scale_upperbound_loss(scale, upperbound, weight):
    scale_x = scale[0]
    scale_y = scale[1]
    upperbound_loss_x = torch.max(torch.abs(scale_x)-upperbound,torch.zeros(len(scale_x)).cuda())
    upperbound_loss_y = torch.max(torch.abs(scale_y)-upperbound,torch.zeros(len(scale_x)).cuda()) 
    return torch.mean(torch.abs(upperbound_loss_x) + torch.abs(upperbound_loss_y))*weight

def get_scale_lowerbound_loss(scale, lowerbound, weight):
    scale_x = scale[0]
    scale_y = scale[1]    
    lowerbound_loss_x = torch.max(lowerbound-scale_x,torch.zeros(len(scale_x)).cuda())    
    lowerbound_loss_y = torch.max(lowerbound-scale_y,torch.zeros(len(scale_x)).cuda())  
    return torch.mean(torch.abs(lowerbound_loss_x) + torch.abs(lowerbound_loss_y))*weight