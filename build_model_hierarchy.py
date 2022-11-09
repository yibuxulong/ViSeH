import io
import os
import os.path
import time
import numpy as np
import torch
import torch.utils.data
from torch.nn import functional as F

# generating features------------
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
        if seq_len == 1:
            predict_list.append(predicts_t[i,:seq_len].unsqueeze(0))
            vector_list.append(hidden_vectors[i,:seq_len].unsqueeze(0))
            new_theta_list.append(theta_list[i,:seq_len].unsqueeze(0))
        else:
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


# launch clustering-----------------------------------------------------------------------------------
def vigilance_check(Match_scores, rhos):
    # M > rho
    diff = Match_scores - rhos
    # if Match > rho - save data in the cluster
    # else create new cluster
    passed_IDs = torch.nonzero(diff > 0).view(-1)
    return passed_IDs


def new_cluster_creation(W, J, L, cluster_label_indicators, data_Assign, data, index, data_label_indicator):
    # J - num of clusters
    J += 1
    # W - weight of cluters
    W[J - 1] = data
    # L - num of data in this cluster
    L[J - 1] = 1
    # Assign - this data belong to which cluster
    data_Assign[index] = J - 1
    
    cluster_label_indicators[J - 1] = data_label_indicator
    return W, J, L, cluster_label_indicators, data_Assign


def old_cluster_update(opt, winnerID, W, L, rhos, data_Assign, data, index):
    # update cluster weights
    # ART
    W[winnerID, :] = opt.art_beta * torch.min(W[winnerID, :], data) + (1 - opt.art_beta) * W[winnerID, :]

    # cluster assignment
    # L - num of data in the cluster
    L[winnerID] += 1
    # Assign - data belong to which cluster
    data_Assign[index] = winnerID

    # increase the rho of the winner (AM-ART)
    rhos[winnerID] = (1 + opt.art_sigma) * rhos[winnerID]

    return W, L, rhos, data_Assign


def clustering_steps(opt, data, data_label_indicators, indexes, networks, data_Assign):
    [W, J, L, rhos, cluster_label_indicators] = networks
    
    for i in range(len(data)):  # for each data item, assign it to a cluster
        # compute the label compatibility - whether the item has the same labels in class & word.
        # same label - 1 , same word - 1
        # only same label = 1 , must have same label and same word(at least one word)
        label_match = torch.sum(data_label_indicators[i] * cluster_label_indicators[:J], 1)
        label_matched_clusterIDs = torch.nonzero(label_match > 0).view(-1)
        if len(label_matched_clusterIDs) == 0:  # if no cluster matched, create a new cluster
            W, J, L, cluster_label_indicators, data_Assign = new_cluster_creation(W, J, L, cluster_label_indicators,data_Assign, data[i], indexes[i],data_label_indicators[i])
        else:  # if matched clusters exist
            intersection = torch.min(data[i], W[label_matched_clusterIDs]).sum(1)
            # M of ART
            Match_scores = intersection / torch.sum(data[i])
            # T of ART
            Choice_scores = intersection / (opt.art_alpha + torch.sum(W[label_matched_clusterIDs], 1))

            # perform vigilance check
            vigilance_passed_cluster_indxes = vigilance_check(Match_scores, rhos[label_matched_clusterIDs])
            
            if len(vigilance_passed_cluster_indxes) == 0:  # no cluster fits the input data
                # create a new cluster
                W, J, L, cluster_label_indicators, data_Assign = new_cluster_creation(W, J, L, cluster_label_indicators, data_Assign, data[i], indexes[i],data_label_indicators[i])
                
            else:  # if the winner exists
                # find the winner cluster
                winner_index = torch.argmax(Choice_scores[vigilance_passed_cluster_indxes])
                winnerID = label_matched_clusterIDs[winner_index]

                # resonance occurs, update the winner
                W, L, rhos, data_Assign = old_cluster_update(opt, winnerID, W, L, rhos, data_Assign, data[i], indexes[i])
                # update the rhos of reset winner clusters (AM-ART)
                winner_choice_value = torch.max(Choice_scores[vigilance_passed_cluster_indxes])
                reset_winnerIDs = \
                    [torch.nonzero((Choice_scores - winner_choice_value) > 0).view(-1)]
                if len(reset_winnerIDs):
                    rhos[reset_winnerIDs] = (1 - opt.art_sigma) * rhos[reset_winnerIDs]

    return [W, J, L, rhos, cluster_label_indicators], data_Assign


def clustering_epoch(opt, epoch, networks, data_Assign, data_loader):
    for batch_idx, (data, data_label_indicators, indexes) in enumerate(data_loader):
        # load network parameters
        # W - Vector*2 (vector, 1-vector) complement codes
        [W, J, L, rhos, cluster_label_indicators] = networks

        # move tensors to GPU
        data = data.cuda()
        # do complement coding
        start_time = time.time()
        # complement codes
        data = torch.cat([data, 1 - data], 1)
        # do clustering
        if epoch == 1 and batch_idx == 0:  # for the first batch, create a cluster using the first data
            # update cluster info
            W[0, :] = data[0, :]  # set cluster weight
            J = 1  # update no. of cluster
            L[0] = 1  # update cluster size
            cluster_label_indicators[0, :] = data_label_indicators[0]
            # Assign this data belong to which cluster
            data_Assign[indexes[0]] = 0
            
            networks, data_Assign = clustering_steps(opt, data[1:], data_label_indicators[1:], indexes[1:],
                                                     [W, J, L, rhos, cluster_label_indicators], data_Assign)
        else:  # for the normal cases, do clustering for the whole batch directly
            networks, data_Assign = clustering_steps(opt, data, data_label_indicators, indexes,
                                                     [W, J, L, rhos, cluster_label_indicators], data_Assign)

        print(
            'Epoch: {} | Processed batch {}/{} | #cluster: {} | time: {:.4f}'.
            format(epoch, batch_idx, len(data_loader), networks[1], time.time() - start_time))
    return networks, data_Assign


def network_init(opt, num_data, num_words, dim_feat):
    # set cluster variables and initialization
    # W, 1-W
    W = torch.zeros([num_data, dim_feat * 2])  # cluster weights; initialize num_cluster = num_data

    # total clusters
    J = 0  # number of clusters

    # L - how many data in this cluster
    L = torch.zeros([num_data], dtype=torch.int32)  # cluster sizes; initialize with num_data
    rhos = opt.art_rho_0 * torch.ones([num_data])  # cluster vigilance values
    # num_class + num_word = 547
    cluster_label_indicators = torch.zeros([num_data, num_words])

    # belong to which cluster(save its index)
    data_Assign = torch.zeros([num_data], dtype=torch.int32) - 1

    W = W.cuda()
    rhos = rhos.cuda()
    return [W, J, L, rhos, cluster_label_indicators], data_Assign


def save_results(opt, epoch, cluster_path, start_time, networks, data_Assign):
    [W, J, L, rhos, cluster_label_indicators] = networks
    W = W[:J].cpu()
    L = L[:J].cpu()
    rhos = rhos[:J].cpu()
    cluster_label_indicators = cluster_label_indicators[:J].cpu()
    np.savez(cluster_path + 'networks_' + str(epoch) + '.npz', W=W, J=J, L=L, rhos=rhos,
             cluster_label_indicators=cluster_label_indicators)
    np.save(cluster_path + 'data_Assign_' + str(epoch) + '.npy', data_Assign)

    print('Epoch: {} completed! {} clusters generated! | Max size: {} | Mean size: {} | time: {:.4f}'.
          format(epoch, J, torch.max(L), torch.mean(L.float()), time.time() - start_time))


def network_clear(opt, networks, data_Assign):
    # clear past results of cluster sizes and cluster assignment of data
    # keep weight and num of cluster
    [W, J, L, rhos, cluster_label_indicators] = networks
    L[:] = 0
    data_Assign[:] = -1
    return [W, J, L, rhos, cluster_label_indicators], data_Assign

