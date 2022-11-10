import argparse

def opt_algorithm():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--dataset', type=str, default= 'food101',help='indicator to dataset')
    # path setting
    parser.add_argument('--result_path', type=str, default= 'result/',help='path to the folder to save results')
    parser.add_argument('--hierarchy_path', type=str, default= 'hierarchy/',help='path to the folder to save hierarchy')
    
    # experiment controls
    parser.add_argument('--net_v', type=str, default= 'resnet18',help='choose network backbone for image channel')
    parser.add_argument('--net_s', type=str, default= 'gru',help='choose network backbone for ingredient channel')
    parser.add_argument('--method', type=str, default= 'global',help='to chose model and dataset')    
    
    # common parameters
    
    parser.add_argument('--batch_size', type=int, default = 64, help='batch size')
    parser.add_argument('--lr', type=float, default = 5e-4, help='learning rate')
    parser.add_argument('--lrd_rate', type=float, default = 0.1, help='decay rate of learning rate')
    parser.add_argument('--lr_decay', type=int, default = 4, help='decay rate of learning rate')
    parser.add_argument('--weight_decay', type=float, default = 1e-3, help='weight decay')

    parser.add_argument('--w_semantic', type=float, default = 5, help='weight of semantic')
    parser.add_argument('--w_visual', type=float, default = 5, help='weight of visual')
    
    # FVSA param
    parser.add_argument('--tsu', type=float, default = 0.8, help='theta of scale upperbound')
    parser.add_argument('--tsl', type=float, default = 0.8, help='theta of scale lowerbound')
    parser.add_argument('--wul', type=float, default = 1., help='weight of upper loss')
    parser.add_argument('--wll', type=float, default = 1., help='weight of lower loss')
    parser.add_argument('--wdl', type=float, default = 5., help='weight of diverse loss')
    parser.add_argument('--waols', type=float, default = 100., help='weight of anti outlier loss shift')
    parser.add_argument('--waolb', type=float, default = 100., help='weight of anti outlier loss bounds')
    parser.add_argument('--wrl', type=float, default = 0., help='weight of rotate loss')
    
    parser.add_argument('--topk', type=int, default = 9,help='top-k classes')
    parser.add_argument('--top_cls', type=int, default = 1,help='top cls for hierarchy')
    parser.add_argument('--top_pos', type=int, default = 5,help='top pos in each seq')
    parser.add_argument('--top_seq', type=int, default = 9,help='top seq, default: the mean of words in dataset')

    # VSHC param
    parser.add_argument('--beta_know', type=float, default = 0.1, help='weight of relation refine')
    
    parser.add_argument('--art_alpha', type=float, default = 1e-2, help='choice parameter of ART')
    parser.add_argument('--art_beta', type=float, default = 5e-1, help='learning parameter of ART')
    parser.add_argument('--art_sigma', type=float, default = 0., help='restraint parameter in AM-ART')
    parser.add_argument('--art_rho_0', type=float, default = 0.85, help='vigilance parameter of ART')
    parser.add_argument('--art_P_T', type=float, default = 0.6, help='decision rate of dominant class')
    parser.add_argument('--art_epoch', type=int, default = 2, help='decision rate of dominant class')
    
    # MMGF param
    parser.add_argument('--beta_relation', type=float, default = 0.2, help='relation weight of gcn')    
    parser.add_argument('--beta_fusion', type=float, default = 0.3, help='weight of decision')
    
    args = parser.parse_args()
    
    return args