import scipy.io as matio
import torch
import torch.utils.data
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim

# Model
class VisualNet(nn.Module):
    def __init__(self, img_encoder, net_type, method, num_cls, dim_feature):
        super(VisualNet, self).__init__()
        self.net_type = net_type
        self.method = method
        self.classifier = nn.Linear(dim_feature, num_cls)

        # utilities
        self._initialize_weights()
        self.relu = nn.LeakyReLU()
        self.encoder_v = img_encoder
        
    def forward(self, x):  # x:image
        x_latent = self.encoder_v(x)
        output = self.classifier(x_latent)
        return output
    def forward_generate(self, x):
        x_latent = self.encoder_v(x)
        output = self.classifier(x_latent)
        return x_latent, output
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                

class word_decoder_STN(nn.Module):
    def __init__(self, CUDA, dim_v, img_encoder):
        super(word_decoder_STN, self).__init__()

        self.CUDA = CUDA
        self.dim_v = dim_v
        
        self.avgpooling = nn.AdaptiveAvgPool2d((1,1))
     
        self.gru = nn.GRUCell(self.dim_v, self.dim_v)

        # create trainable initial hidden state
        h0_de = torch.zeros([1, self.dim_v])
        if self.CUDA:
            h0_de = h0_de.cuda()
        self.h0_de = nn.Parameter(h0_de, requires_grad=True)
        
        self.STN = STN(self.dim_v)
        
    def forward(self, x_fm, max_seq):
        h_list = []
        theta_list = []
        current_hidden = self.h0_de.repeat([x_fm.shape[0], 1])
        
        for i in range(0, max_seq):  # for each of the max_seq for decoder
            #get current input from STN
            current_input, theta = self.getCurInput(x_fm, current_hidden)
            theta_list.append(theta)
            # perform a gru op
            current_hidden = self.gruLoop(current_input, current_hidden)      
            h_list.append(current_hidden)
           
        return torch.stack(h_list, 0), torch.stack(theta_list, 0)

    def getCurInput(self, x_fm, current_hidden):
        patch, theta = self.STN.gru_forward(x_fm, current_hidden)
        x_latent = self.avgpooling(patch).view((patch.shape[0], -1))
        return x_latent, theta  
    
    def gruLoop(self, current_input, prev_hidden):
        # use it to avoid a modification of prev_hidden with inplace operation
        return self.gru(current_input, prev_hidden)
                
                
class STN(nn.Module):
    def __init__(self, dim_v):
        super(STN, self).__init__()

        # regression network for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(dim_v*2, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 6)
        )
        # initialize non-pretrained parameters
        self._initialize_weights()   

        # Initialize the weights/bias of the last layer in fc_loc to be identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([0.5, 0, 0, 0, 0.5, 0], dtype=torch.float)) 
        # transform x_fm to feature vector
        self.avgpooling = nn.AdaptiveAvgPool2d((1,1))

    def gru_forward(self, x_fm, hidden):  # x:image feature maps
        x_latent = self.avgpooling(x_fm).view((x_fm.shape[0], -1))
        theta = self.fc_loc(torch.cat([x_latent, hidden], 1))
        theta = theta.view(-1, 2, 3)
        
        grid = F.affine_grid(theta, x_fm.size())
        x_fm = F.grid_sample(x_fm, grid)
        return x_fm, theta
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

#-----------------task 2-----------------
class FVSA(nn.Module):
    def __init__(self, CUDA, img_encoder, dim_v, num_words):
        super(FVSA, self).__init__()
        
        self.CUDA = CUDA
        self.num_words = num_words
        self.dim_v = dim_v
        
        # classifiers
        self.word_classifier = nn.Linear(dim_v, num_words)
        
        # initialize
        self._initialize_weights()
        
        # network for image encoder
        self.img_encoder = img_encoder
        # word decoder
        self.word_decoder = word_decoder_STN(CUDA, self.dim_v, self.img_encoder)

    def forward(self, x, max_seq_batch):  # x:image, max_seq_batch: max_seq of current batch
        # get img feature maps
        x_fm = self.img_encoder(x)
        # decode to hidden vectors (seq, batch, latent_len) for individual words
        hidden_vectors, theta_list = self.word_decoder(x_fm, max_seq_batch)
   
        # compute word prediction (seq, batch, latent_len) -> (batch, seq, num_words)
        predicts_t = self.word_classifier(hidden_vectors.transpose(0, 1))

        return predicts_t, x_fm, hidden_vectors, theta_list
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  



class VSRF(nn.Module):
    def __init__(self, patch2word, dim_v, num_cls, num_words, top, cluster_weights, cluster_label_indicators, valid_cluster_class, feature_max, feature_min):
        super(VSRF, self).__init__()
        [top_cls, top_seq, top_pos, topk] = top
        self.num_words = num_words
        self.num_cls = num_cls
        self.dim_v = dim_v
        self.topk = topk
        self.top_cls = top_cls
        self.top_seq = top_seq
        self.top_pos = top_pos
        self.feature_max = feature_max
        self.feature_min = feature_min
        self.cluster_weights = cluster_weights
        self.cluster_label_indicators = cluster_label_indicators
        self.valid_cluster_class = valid_cluster_class
             
        # network for image encoder
        self.patch2word = patch2word
        
    def forward(self, x, max_seq_batch):  # use label for construct adj in training
        predicts_t, x_fm, feature_s, _ = self.patch2word(x, max_seq_batch)
        feature_v = nn.AdaptiveAvgPool2d((1,1))(x_fm)
        feature_v = feature_v.view(feature_v.shape[0],feature_v.shape[1])
        # get word & class decisions by filtering word prediction using the word cluster hierarchy
        decision_words, decision_classes_topk = self.getKnowledgeFiltering(predicts_t, feature_s.transpose(0, 1))
        
        return predicts_t.cpu(), feature_v, decision_words, decision_classes_topk
    
    def ART_normalizer(self, hidden_vectors):
        scaled_vectors = (hidden_vectors - self.feature_min) / (self.feature_max - self.feature_min)
        return torch.cat([scaled_vectors,1-scaled_vectors], 2)
    
    def getKnowledgeFiltering(self, predicts_t, hidden_vectors):
        bach_size = predicts_t.shape[0]
        topk_pre_words = torch.topk(predicts_t, self.top_pos)[1]
        topk_pre_words_predicts = torch.topk(predicts_t, self.top_pos)[0]
        #normalize the hidden vectors via min-max normalizer derived from ART clusters
        hidden_vectors = self.ART_normalizer(hidden_vectors)
        
        #save the final decisions for words and classes
        decision_words = []
        decision_classes = []
        decision_classes_topk = []

        #for the predictions for each word, make the final decision using the cluster hierarchy
        for i in range(bach_size):
            
            #save the prediction decision for each word 
            cluster_based_decisions = []
            final_word_decisions = []
            
            #save the class decisions from the best-matching clusetrs of each word
            class_similarity_all = []
            class_labels_all = []
            
            #process each valid word in seq
            for j in range(self.top_seq): 
                #obtain the probs for each prediction using cluster hierarchy and make a final decision
                topk_predicts = topk_pre_words_predicts[i,j]  
                Topk_cluster_similarity_all = []
                Topk_class_labels_all = []
                
                #for each prediction, pick the best matching clusters
                for k in range(self.top_pos):
                    wordID = topk_pre_words[i,j,k] # predicted word of pos_k in seq_j
                    self.word_label_indicator = torch.zeros(self.num_words).cuda()     
                    self.word_label_indicator[wordID] = 1
                    
                    #find valid clusters
                    indicators = (self.word_label_indicator * self.cluster_label_indicators).sum(1)
                    valid_clusterIDs = torch.where(indicators > 0)[0]
                   
                    if len(valid_clusterIDs)>0:
                        #perform similarity measure of ART; pick the best matching one as similarity
                        intersection = torch.min(hidden_vectors[i,j], self.cluster_weights[valid_clusterIDs]).sum(1) 
                        match_scores = intersection / torch.sum(hidden_vectors[i,j])
                        choice_scores = intersection / (0.01 + torch.sum(self.cluster_weights[valid_clusterIDs], 1))  # alpha = 0.1
                        unification = torch.max(hidden_vectors[i,j], self.cluster_weights[valid_clusterIDs]).sum(1)
                        similarity = intersection / unification
                        
                        num_cluster_cadidates = min(self.top_cls, len(valid_clusterIDs))
                        
                        sorted_similarity = torch.topk(similarity, similarity.shape[0])[1]
                        topk_cluster_idx = sorted_similarity[:num_cluster_cadidates] 
                        topk_clusterIDs = valid_clusterIDs[topk_cluster_idx]

                        topk_cluster_class_label = self.valid_cluster_class[topk_clusterIDs]

                        topk_cluster_similarities = similarity[topk_cluster_idx]
                        #compute the probability for this prediction via cluster hierarchy
                        Topk_cluster_similarity_all.append(topk_cluster_similarities)  

                        #save the class prediction according to the cluster hierarchy
                        Topk_class_labels_all.append(topk_cluster_class_label)
                    else:
                        intersection = torch.min(hidden_vectors[i,j], self.cluster_weights).sum(1)
                        match_scores = intersection / torch.sum(hidden_vectors[i,j])
                        choice_scores = intersection / (0.01 + torch.sum(self.cluster_weights, 1))  # alpha = 0.1

                        unification = torch.max(hidden_vectors[i,j], self.cluster_weights).sum(1)
                        similarity = intersection / unification #(#cluster) jiaoji / heji

                        #get the class label of the top-1 clusters
                        num_cluster_cadidates = self.top_cls#   
                        
                        sorted_similarity = torch.topk(similarity, similarity.shape[0])[1]
                        topk_cluster_idx = sorted_similarity[:num_cluster_cadidates] # num: top_cls
                        topk_clusterIDs = topk_cluster_idx 

                        topk_cluster_class_label = self.valid_cluster_class[topk_clusterIDs]

                        topk_cluster_similarities = similarity[topk_cluster_idx]
            
                        #compute the probability for this prediction via cluster hierarchy
                        Topk_cluster_similarity_all.append(topk_cluster_similarities)  

                        #save the class prediction according to the cluster hierarchy
                        Topk_class_labels_all.append(topk_cluster_class_label)
                
                # gather top_pos class-predictions and similaritys of seq_j, and decision the class
                Topk_cluster_similarity_all = torch.cat(Topk_cluster_similarity_all)
                # chosing top1 for single-label dataset, chosing topk for multi-label dataset
                cluster_based_decisions.append(torch.topk(Topk_cluster_similarity_all,1)[1].item()) 
                
                topk_predicts = topk_predicts.unsqueeze(1).repeat(1,self.top_cls).reshape(-1)
                final_word_probs = Topk_cluster_similarity_all * topk_predicts
                winner_idx = torch.topk(final_word_probs,1)[1].item()
                winnerID = topk_pre_words[i,j,winner_idx]
                final_word_decisions.append(winnerID.item())
                
                #make the class prediction using the class label info for each word
                class_similarity_all.append(final_word_probs.max())
                class_labels_all.append(Topk_class_labels_all[winner_idx][0].long().item())

            cluster_predict_words = topk_pre_words[i,torch.arange(len(cluster_based_decisions)),torch.tensor(cluster_based_decisions)]
            decision_words.append(final_word_decisions)
            
            # save class predictions
            
            class_similarity_all = torch.tensor(class_similarity_all)
            cluster_predict_class_topk = torch.tensor(class_labels_all)[torch.topk(class_similarity_all, self.topk)[1]].int().tolist()
            decision_classes_topk.append(cluster_predict_class_topk)

        #return the final decisions on words and food classes    
        return torch.tensor(decision_words), torch.tensor(decision_classes_topk)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                

class CAGL(nn.Module):
    def __init__(self, dim_v, num_cls, num_words, topk, beta_know, beta_relation):
        super(CAGL, self).__init__()
        
        self.num_words = num_words
        self.num_cls = num_cls
        self.dim_v = dim_v
        self.topk = topk
        self.beta_know = beta_know
        self.beta_relation = beta_relation
        
        # classifiers
        self.class_classifier = nn.Linear(self.dim_v+self.dim_v, num_cls)
        self.embed_words = nn.Embedding(self.num_words+1, self.dim_v)
        self.embed_fuse = nn.Linear(self.topk, 1)
        
        # initialize
        self._initialize_weights()
        self.softmax = nn.Softmax(1)
    
    def forward(self, predicts_t, feature_v, decision_words):
        # top_n predicts
        predicts_topk_id, predicts_topk_pre = self.refine_word_pre(predicts_t, decision_words, self.topk, self.beta_know)
        word_embed = self.embed_words(predicts_topk_id)
        visual_embed = feature_v.unsqueeze(1).repeat(1,self.topk,1)
        mix_embed = torch.cat((word_embed, visual_embed), 2)
        #adj = self.compute_adj(predicts_t.shape[0], self.beta_relation)
        adj = self.compute_refined_adj(predicts_t.shape[0], self.beta_relation, predicts_topk_id, decision_words)
        
        mix_embed_gcn = torch.bmm(adj, mix_embed)
        mix_embed_fuse = self.embed_fuse(mix_embed_gcn.transpose(1,2)).squeeze()
        
        predicts_v = self.class_classifier(mix_embed_fuse)
        
        return mix_embed_fuse, predicts_v
        
        
    def minmax_oprtator(self, data):
        min = torch.min(data, dim=2)[0].unsqueeze(2).expand(data.shape[0],data.shape[1],data.shape[2]) 
        max = torch.max(data, dim=2)[0].unsqueeze(2).expand(data.shape[0],data.shape[1],data.shape[2])  
        return (data - min)/(max-min)
    
    def refine_word_pre(self, model_pre, vsrf_pre, k, beta):
        model_pre_minmax = self.minmax_oprtator(model_pre)
        model_pre = torch.max(model_pre_minmax, dim=1)[0]
        
        vsrf_pre = torch.sum(F.one_hot(vsrf_pre.long(), self.num_words), dim=1).float()
        
        predicts_refine = (1.-beta) * self.softmax(model_pre) + self.softmax(vsrf_pre) * beta
        
        predicts_topk_id = torch.sort(predicts_refine, dim=1, descending=True)[1][:,:k] #(64,25)
        predicts_topk_pre = torch.sort(predicts_refine, dim=1, descending=True)[0][:,:k] #(64,25)
        
        return predicts_topk_id, predicts_topk_pre
    
    def compute_adj(self, batch_size, beta_relation):
        adj_init = torch.eye((self.topk)).unsqueeze(0).repeat(batch_size, 1, 1)
        adj_init = beta_relation*(torch.ones((self.topk,self.topk)).unsqueeze(0).repeat(batch_size,1,1)-adj_init)+adj_init
        adj_deg = torch.sum(adj_init, dim=2)
        adj_norm = (adj_deg**-1).unsqueeze(2) * adj_init
        adj_norm[torch.isnan(adj_norm)] = 0.
        return adj_norm.cuda()
    
    def compute_refined_adj(self, batch_size, beta_relation, predicts_topk_id, decision_words):
        predicts_topk_id_onehot = torch.max(F.one_hot(predicts_topk_id, self.num_words),1)[0]
        decision_words_onehot = torch.max(F.one_hot(decision_words.long(), self.num_words),1)[0]
        decision_inter = (predicts_topk_id_onehot * decision_words_onehot).float().unsqueeze(2)
        decision_adj = torch.bmm(decision_inter, decision_inter.transpose(1,2)) * self.beta_relation
        decision_adj_init = torch.zeros((batch_size, self.topk, self.topk))
        for i in range(batch_size):
            decision_adj_init[i] = decision_adj[i][predicts_topk_id[i]][:, predicts_topk_id[i]]
        
        adj_init = torch.eye((self.topk)).unsqueeze(0).repeat(batch_size, 1, 1)
        adj_init = beta_relation*(torch.ones((self.topk,self.topk)).unsqueeze(0).repeat(batch_size,1,1)-adj_init)+adj_init
        
        adj_init += decision_adj_init
        adj_deg = torch.sum(adj_init, dim=2)
        adj_norm = (adj_deg**-1).unsqueeze(2) * adj_init
        adj_norm[torch.isnan(adj_norm)] = 0.
        return adj_norm.cuda()   
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                                         

class FeatureFusion(nn.Module):
    def __init__(self, dim_v, num_class):
        super(FeatureFusion, self).__init__()
        
        self.linearg2g = nn.Linear(dim_v, dim_v)
        self.linearl2g = nn.Linear(dim_v*2, dim_v)
        
        self.classifier = nn.Linear(dim_v, num_class)
        self.beta = 0.6
        self._initialize_weights()
        

    def forward(self, feature_global, feature_cagl):
        feature_global = self.linearg2g(feature_global)
        feature_cagl = self.linearl2g(feature_cagl)
        feature_fusion = ((1.-self.beta) * feature_global) + (feature_cagl * self.beta)
        output = self.classifier(feature_fusion)
        return feature_fusion, output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                
                
def select_visual_network(img_net_type, image_size,opt,return_fm=False):
    if img_net_type == 'resnet50':
        from model.resnet import resnet50
        img_encoder = resnet50(image_size, pretrained=True, return_fm=return_fm)
    elif img_net_type == 'resnet18':
        from model.resnet import resnet18
        img_encoder = resnet18(image_size, pretrained=True, return_fm=return_fm)
    elif img_net_type == 'vit':
        import model.model_ViT as ViT
        img_encoder = ViT.VisionTransformer(image_size=(384, 384),patch_size=(32, 32),return_fm=return_fm)
    else:
        assert 1 < 0, 'Please indicate backbone network of image channel with any of resnet50/vgg19bn/wrn/wiser'

    return img_encoder




def get_updateModel_v_vit(model, path):
    pretrained_dict = torch.load(path, map_location='cpu')['state_dict']
    model_dict = model.state_dict()
    shared_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(shared_dict)
    model.load_state_dict(model_dict)

    return model

def get_updateModel(model, path):
    pretrained_dict = torch.load(path, map_location='cpu')
    model_dict = model.state_dict()
    shared_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items() if k.replace('module.', '') in model_dict}
    model_dict.update(shared_dict)
    model.load_state_dict(model_dict)

    return model



def build(CUDA, opt):
    # network for image channel
    if opt.net_v=='resnet18':
        dim_feat_v = 512
    elif opt.net_v=='resnet50':
        dim_feat_v = 2048
    elif opt.net_v=='vit':
        dim_feat_v = 768
        
    # build model
    if opt.method=='global':
        print(opt.net_v)
        encoder_v = select_visual_network(opt.net_v, opt.size_img, opt)
        # encoder_v = get_updateModel_v_vit(encoder_v, 'PATH_OF_PRETRAINED_ViT') # for global modal training
        model = VisualNet(encoder_v, opt.net_v, opt.method, opt.num_cls, dim_feat_v)  
    elif opt.method=='fvsa':
        encoder_v = select_visual_network(opt.net_v, opt.size_img, opt, return_fm=True)
        model = FVSA(CUDA, encoder_v, dim_feat_v, opt.num_words)
    elif opt.method=='vsrf':
        encoder_v = select_visual_network(opt.net_v, opt.size_img, opt, return_fm=True)
        model_fvsa = FVSA(CUDA, encoder_v, dim_feat_v, opt.num_words)
        # load model in task2
        model_fvsa = get_updateModel(model_fvsa, opt.path_fvsa)
        model = VSRF(model_fvsa, dim_feat_v, opt.num_cls, opt.num_words, [opt.top_cls, opt.top_seq, opt.top_pos, opt.topk], opt.cluster_weights, opt.cluster_label_indicators, opt.valid_cluster_class, opt.feature_max, opt.feature_min)
    elif opt.method=='cagl':
        model = CAGL(dim_feat_v, opt.num_cls, opt.num_words, opt.topk, opt.beta_know, opt.beta_relation)
    elif opt.method=='fusion':
        model = FeatureFusion(dim_feat_v, opt.num_cls)
    if CUDA:        
        model = model.cuda()
    return model

