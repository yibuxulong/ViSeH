import io
import scipy.io as matio
import numpy as np
from PIL import Image
import torch
import torch.utils.data
import ipdb


def default_loader(image_path):
    return Image.open(image_path).convert('RGB')


class dataset_for_classification(torch.utils.data.Dataset):
    def __init__(self, opt, mode, transform):
        # image channel data
        if mode == 'train':
            img_path_file = opt.path_data + 'train_images.txt'
            with io.open(opt.path_data + 'train_labels.txt', encoding='utf-8') as file:
                labels = file.read().split('\n')[:-1]
            words = matio.loadmat(opt.path_data + 'ingredient_all_feature.mat')['ingredient_all_feature']
            indexVectors = matio.loadmat(opt.path_data + 'ingredient_all_feature.mat')['ingredient_all_feature']
        elif mode == 'test':
            img_path_file = opt.path_data + 'test_images.txt'
            with io.open(opt.path_data + 'test_labels.txt', encoding='utf-8') as file:
                labels = file.read().split('\n')[:-1]
            words = matio.loadmat(opt.path_data + 'ingredient_all_feature.mat')['ingredient_all_feature']   
        self.img_label = np.array(labels, dtype=int)
        with io.open(img_path_file, encoding='utf-8') as file:
            path_to_images = file.read().split('\n')[:-1]
        
        self.dataset = opt.dataset
        self.path_img = opt.path_img
        
        self.path_to_images = path_to_images
        self.transform = transform
        self.loader = default_loader
        words = words.astype(np.float32)
        
        self.words = words

    def __getitem__(self, index):
        # get image matrix and transform to tensor
        path = self.path_to_images[index]
        img = self.loader(self.path_img + path + '.jpg')
        label = self.img_label[index]
        if self.transform is not None:
            img = self.transform(img)
        # get index vector for gru input
        words = self.words[label]
        return [img, words], label
    
    def __len__(self):
        return len(self.path_to_images)

class dataset_for_art(torch.utils.data.Dataset):
    def __init__(self, num_words, data_path = None):
        
        hidden_vector_scaled = np.load(data_path+'/scaled_hidden_vectors.npy')
        hidden_vector_wordIDs = np.load(data_path+'/hidden_vector_wordIDs.npy').astype(np.long) # (60w,) patch-ingre

        self.vectors = hidden_vector_scaled  # (627075,2048)
        self.num_words = num_words
        self.wordIDs = hidden_vector_wordIDs # (627075,)

    def __getitem__(self, index):
        # get hidden vector       
        vector = torch.from_numpy(self.vectors[index])
        wordIDs = self.wordIDs[index]
        
        label_indicator = torch.zeros([self.num_words], dtype =torch.float32)
        # ingreID = 1
        label_indicator[wordIDs] = 1
        
        return [vector, label_indicator, index]

    def __len__(self):
        return len(self.wordIDs)

class dataset_for_cagl(torch.utils.data.Dataset):
    def __init__(self, label, feature_v, word_predicts, decision_classes_topk, decision_words):
        self.label = label.long()
        self.feature_v = feature_v
        self.word_predicts = word_predicts
        self.decision_classes_topk = decision_classes_topk
        self.decision_words = decision_words
    def __getitem__(self, index): 
        label = self.label[index]
        feature_v = self.feature_v[index]
        word_predicts = self.word_predicts[index]
        decision_classes_topk = self.decision_classes_topk[index]
        decision_words = self.decision_words[index]
        
        return [feature_v, word_predicts, decision_classes_topk, decision_words], label
    
    def __len__(self):
        return len(self.label)

    
class dataset_for_fusion(torch.utils.data.Dataset):
    def __init__(self, feature_global, feature_local, label):
        self.feature_global = feature_global
        self.feature_local = feature_local
        self.label = label
    def __getitem__(self, index): 
        label = self.label[index]
        feature_global = self.feature_global[index]
        feature_local = self.feature_local[index]
        
        return [feature_global, feature_local], label
    
    def __len__(self):
        return len(self.label)
