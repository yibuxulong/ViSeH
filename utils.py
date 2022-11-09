import torch
import torch.nn as nn
import shutil
import os
import numpy as np
from sklearn.metrics import average_precision_score

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0) # batch_size

        _, pred = output.topk(maxk, 1, True, True) # pred: (batch_size, maxk)
        pred = pred.t() # pred: (maxk, batch_size)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append(correct_k.mul_(100.0 / batch_size))
        return results
    


def accuracy_hit(output, target, num_cls, topk=(1,)):
    output = output.cpu()
    target = target.cpu()
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0) # batch_size

        _, pred = output.topk(maxk, 1, True, True) # pred: (batch_size, maxk)
        pred = pred.t() # pred: (maxk, batch_size)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            if k==1:
                hit_k = correct[:k].reshape(-1).float()
                hit_cls = np.zeros(num_cls)
                ind = 0
                for i in range(len(hit_k)):
                    if hit_k[i] == 1:
                        hit_cls[target[i]]+=1
            results.append(correct_k.mul_(1.0 / batch_size))
        return results, hit_cls


def prepare_intermediate_folders(path):
    if not os.path.exists(path):
        os.makedirs(path)

def para_name(opt):
    name_para = 'datset={}~net_v={}~net_s={}~method={}~lr={}'.format(
    opt.dataset,
    opt.net_v,
    opt.net_s,
    opt.method,
    opt.lr
    )
        
    return name_para

class AverageMeter(object):
    
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)