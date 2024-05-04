import os
import pdb
import torch
import tqdm
# from models import DEVICE
from tap import Tap
from typing import List, Union
from typing import Optional, Sequence
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

DEVICE='cuda'
class EvalArgs(Tap):
    ckpt_path: str = None
    ckpt_list: List[str] = None  # DEns 
    gde_ckpt_list: List[str] = None  # GDE 
    dropout_layers: List[int] = None  # GDE 
    uq_name = 'vanilla'
    save_results: int = 1


def get_logits_labels(model,loader,anchors=None,n_anchors=None):
    model.eval()
    preds = []
    targets = []
    unqs=[]
    with torch.no_grad(): 
        for batch in tqdm.tqdm(loader,disable=True):
            if anchors is not None and n_anchors > 1:
                """
                By passing return_std, the model will return the calibrated logits! 
                """
                out,unq = model(batch.to(DEVICE),
                    anchors=anchors, 
                    n_anchors=n_anchors,
                    return_std=True)
                unqs.append(unq)
            elif anchors is not None and n_anchors == 1:
                """
                By passing return_std, the model will return the calibrated logits! 
                """
                out = model(batch.to(DEVICE),
                    anchors=anchors, 
                    n_anchors=n_anchors,
                    return_std=False)
            else:    
                out = model(batch.to(DEVICE))
            labeled = batch.y == batch.y
            # preds.append(out[labeled])
            # targets.append(batch.y[labeled])
            preds.append(out)
            targets.append(batch.y[labeled])

    preds = torch.cat(preds,dim=0).to('cpu') 
    targets = torch.cat(targets,dim=0).to('cpu')
    if anchors is not None and n_anchors > 1:
        unqs = torch.cat(unqs,dim=0).to('cpu')
        return preds, targets,unqs
    return preds,targets

from GOOD.utils.train import nan2zero_get_mask
to_np = lambda x: x.data.to('cpu').numpy()
def get_net_results(net, loader, config, in_dist=False, t=1,device='cpu',split='val'):
    logits = []
    confidence = []
    correct = []
    labels = []
    net.eval()
    with torch.no_grad():
        for batch_idx,data in enumerate(loader):
              
            data = data.to(device)
            mask, _= nan2zero_get_mask(data, split, config)
            target = data.y[mask]
            output = net(data)[mask] #, edge_weight=None, ood_algorithm=None)
            logits.extend(to_np(output).squeeze())

            confidence.extend(to_np(F.softmax(output/t, dim=1).max(1)[0]).squeeze().tolist())
            if in_dist:
                pred = output.data.max(1)[1]
                correct.extend(pred.eq(target).to('cpu').numpy().squeeze().tolist())
                labels.extend(target.to('cpu').numpy().squeeze().tolist())
    return logits, confidence, correct, labels

def get_results_summary(model,loader,cal_metric_l1,cal_metric_l2, cal_metric_max, acc_metric,n_anchors=None, anchors=None,uq_name='vanilla',t=torch.Tensor([1])):
    if anchors is not None and n_anchors > 1:
        preds, targets,unqs = get_logits_labels(model=model,
            loader=loader,
            anchors=anchors,
            n_anchors=n_anchors)
    elif anchors is not None and n_anchors == 1:
        preds, targets= get_logits_labels(model=model,
            loader=loader,
            anchors=anchors,
            n_anchors=n_anchors)
        unqs = None
    else:
        if uq_name.upper() in['VANILLA','TEMP','TS','ETS','VS',"IRM", "DIRICHLET","SPLINE","ORDERINVARIANT"]:
            preds, targets = get_logits_labels(model=model, loader=loader,anchors=anchors,
            n_anchors=n_anchors)
        elif uq_name == 'mcd':
            preds, targets= get_logits_labels_mcd(net=model, loader=loader,dropout_layers=model.dropout_layers)
        
        else: 
            print("INVALID BASELINE; EXITING")
            exit() 
        preds= preds.cpu()
        preds = preds / t.cpu()
        targets = targets.cpu()
        unqs = None

    # one_hot = torch.zeros(len(targets), 2)
    # one_hot.scatter_(1, targets.unsqueeze(1).long(), 1)
    acc, cal_err_l1, cal_err_l2, cal_err_max = torch.Tensor([-1]), torch.Tensor([-1]), torch.Tensor([-1]), torch.Tensor([-1])
    if targets.dim() == 1 and targets.max() == 1:
        targets = targets.unsqueeze(1)
    else:
        pass
    try:
        cal_err_l1 = cal_metric_l1(preds,targets)
    except:
        pass
    try:
        cal_err_l2 = cal_metric_l2(preds,targets)
    except:
        pass
    try: 
        cal_err_max = cal_metric_max(preds,targets)
    except: 
        pass
    try:
        acc = acc_metric(preds,targets)
    except: 
        pass
    stat_dict = {
        'acc':acc,
        'cal_err_l1':cal_err_l1,
        'cal_err_l2':cal_err_l2,
        'cal_err_max':cal_err_max,
        'preds':preds,
        'targets':targets,
        'unqs':unqs
    }
    return stat_dict

def get_logits_labels_wrapper(model,loader,n_anchors=None, anchors=None,uq_name='vanilla',t=torch.Tensor([1])):
    if anchors is not None and n_anchors > 1:
        preds, targets,unqs = get_logits_labels(model=model,
            loader=loader,
            anchors=anchors,
            n_anchors=n_anchors)
    elif anchors is not None and n_anchors == 1:
        preds, targets= get_logits_labels(model=model,
            loader=loader,
            anchors=anchors,
            n_anchors=n_anchors)
        unqs = None
    else:
        if uq_name == 'vanilla' or uq_name == 'temp':
            preds, targets = get_logits_labels(model=model, loader=loader,anchors=anchors,
            n_anchors=n_anchors)
        elif uq_name == 'mcd':
            preds, targets= get_logits_labels_mcd(net=model, loader=loader,dropout_layers=model.dropout_layers)
        
        else: 
            print("INVALID BASELINE; EXITING")
        
        preds= preds.cpu()
        preds = preds / t.cpu()
        targets = targets.cpu()
        unqs = None

    return preds, targets 

def get_logits_labels_dens(model,loader,ckpt_list,anchors=None,n_anchors=None):
    model.eval()
    pred_list = []
    for enum,ckpt_p in enumerate(ckpt_list): 
        ckpt = torch.load(ckpt_p)
        model.encoder.load_state_dict(ckpt["encoder"])
        model.classifier.load_state_dict(ckpt["classifer"])
        model.eval()
        # p, t = get_logits_labels(model=model, loader=loader)
        p, t,_ = get_logits_labels(model=model, loader=loader,anchors=anchors,n_anchors=n_anchors)
        pred_list.append(p)

    preds = torch.stack(pred_list, dim=0).mean(dim=0)
    targets = t #there is no shuffling so we just take the last "t".

    #Restore model to original ckpt
    ckpt = torch.load(ckpt_list[0])
    model.encoder.load_state_dict(ckpt["encoder"])
    model.classifier.load_state_dict(ckpt["classifer"])
    model.eval()

    return preds,targets

def get_logits_labels_duqdens(model,loader,ckpt_list,anchors=None,n_anchors=None,return_unqs=False):
    model.eval()
    pred_list = []
    unqs_list = []
    for enum,ckpt_p in enumerate(ckpt_list): 
        ckpt = torch.load(ckpt_p)
        model.net.encoder.load_state_dict(ckpt["encoder"])
        model.net.classifier.load_state_dict(ckpt["classifer"])
        model.eval()
        # p, t = get_logits_labels(model=model, loader=loader)
        p, t,unqs = get_logits_labels(model=model, loader=loader,anchors=anchors[enum],n_anchors=n_anchors)
        pred_list.append(p)
        unqs_list.append(unqs)

    preds = torch.stack(pred_list, dim=0).mean(dim=0)
    unqs = torch.stack(unqs_list, dim=0).mean(dim=0)
    targets = t #there is no shuffling so we just take the last "t".

    #Restore model to original ckpt
    ckpt = torch.load(ckpt_list[0])
    model.net.encoder.load_state_dict(ckpt["encoder"])
    model.net.classifier.load_state_dict(ckpt["classifer"])
    model.eval()
    if return_unqs:
        return preds, targets, unqs
    return preds,targets


def get_logits_labels_mcd(net, loader,dropout_layers, num_samples=10):
    # First: collect all the logits and labels for the validation set
    logits_list = []
    labels_list = []
    net.eval()
    for l in dropout_layers:
        net.encoder.dropouts[l].train()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            labeled = batch.y == batch.y
            labels_list.append(batch.y[labeled])
            pred_list = []
            for _ in range(num_samples):
                logits = net(batch)
                pred_list.append(logits)

            pred_list = torch.stack(pred_list)
            mu = torch.mean(pred_list,dim=0) #Batch-size, Num Classes (Probits) 
            # std = torch.std(pred_list,dim=0) #Batch-size, Num Classes (Probits)
            logits_list.append(mu[labeled])
    logits = torch.cat(logits_list).to(DEVICE)
    labels = torch.cat(labels_list).to(DEVICE)
    return logits, labels

def compute_temp(net, loader,temp=1,use_binary=False):
    # First: collect all the logits and labels for the validation set
    logits_list = []
    labels_list = []
    nll_criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    if use_binary:
        nll_criterion = torch.nn.BCELoss().to(DEVICE)
    net.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = net(batch)
            logits_list.append(logits)
            labels_list.append(batch.y)
    logits = torch.cat(logits_list).to(DEVICE)
    labels = torch.cat(labels_list).to(DEVICE)
    # pdb.set_trace() 
    # Calculate NLL and ECE before temperature scaling
    if use_binary:
        sig = torch.nn.Sigmoid()
        before_temperature_nll = nll_criterion(sig(logits), labels).item()
    else:
        before_temperature_nll = nll_criterion(logits, labels).item()
    print('Before temperature - NLL: %.3f' % (before_temperature_nll))

   # Next: optimize the temperature w.r.t. NLL
    temperature = torch.nn.Parameter(torch.ones(1).to(DEVICE))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def temperature_scale(logits):
        return torch.div(logits, temperature)

    def eval():
        optimizer.zero_grad()
        if use_binary:
            loss = nll_criterion(temperature_scale(sig(logits)), labels)
        else:
            loss = nll_criterion(temperature_scale(logits), labels)
        loss.backward()
        return loss
    optimizer.step(eval)

    # Calculate NLL and ECE after temperature scaling
    if use_binary:
        after_temperature_nll = nll_criterion(temperature_scale(sig(logits)), labels).item()
    else:
        after_temperature_nll = nll_criterion(temperature_scale(logits), labels).item()
    print('Optimal temperature: %.3f' % temperature.item())
    print('After temperature - NLL: %.3f' % (after_temperature_nll))
    return temperature.detach().to(DEVICE)


import numpy as np
from pathlib import Path
# Run at console -> python -c 'from src.data.data_utils import *; generate_node_to_nearest_training("Cora", "5_3f_85")'
def generate_node_to_nearest_training(dataset_name: str, data, bfs_depth = 10):

        dist_to_train = torch.ones(data.num_nodes) * bfs_depth
        dist_to_train = shortest_path_length(data.edge_index, data['train_mask'], bfs_depth)
        split_file = '/p/lustre3/trivedi1/Summer22/GraphUQ/dist_to_train/{}.npy'.format(dataset_name) 
        np.save(split_file, dist_to_train)

def load_node_to_nearest_training(dataset_name: str, data):
    split_file = '/p/lustre3/trivedi1/Summer22/GraphUQ/dist_to_train/{}.npy'.format(dataset_name) 
    if not os.path.isfile(split_file):
        generate_node_to_nearest_training(dataset_name, data)
    return torch.from_numpy(np.load(split_file)).to(DEVICE)

def shortest_path_length(edge_index, mask, max_hop):
    """
    Return the shortest path length to the mask for every node
    """
    dist_to_train = torch.ones_like(mask, dtype=torch.long, device=mask.device) * torch.iinfo(torch.long).max
    seen_mask = torch.clone(mask)
    for hop in range(max_hop):
        current_hop = torch.nonzero(mask)
        dist_to_train[mask] = hop
        next_hop = torch.zeros_like(mask, dtype=torch.bool, device=mask.device)
        for node in current_hop:
            node_mask = edge_index[0,:]==node
            nbrs = edge_index[1,node_mask]
            next_hop[nbrs] = True
        hop += 1
        # mask for the next hop shouldn't be seen before
        mask = torch.logical_and(next_hop, ~seen_mask)
        seen_mask[next_hop] = True
    return dist_to_train 

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
