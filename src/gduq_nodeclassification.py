import os
import time
import numpy as np
import pdb
import tqdm

from tap import Tap
from pprint import pprint
from munch import Munch 
from munch import munchify

import torch.nn as nn
import torchvision.transforms as transforms
import torch
import torchvision.models as models

from GOOD.data import load_dataset, create_dataloader
from GOOD.networks.models.GINs import GINEncoder
from GOOD.networks.models.GCNs import GCNEncoder
from GOOD.networks.models.GINvirtualnode import vGINEncoder
from GOOD.utils.args import args_parser
from GOOD.utils.config_reader import load_config, args2config, process_configs
from torchmetrics.classification import BinaryCalibrationError, MulticlassCalibrationError,BinaryAccuracy, MulticlassAccuracy
from GOOD.utils.train import nan2zero_get_mask
from torch.optim.lr_scheduler import MultiStepLR 
from utils import dUQArgs, DEVICE, count_parameters, get_results_summary_node, test_node
from models.encoders import GCNEncoder
from models.gduq_models import baseModelNode, GraphANTNode 

def main():
    '''
    Setup Model
    '''
    
    args = args_parser()
    config, duplicate_warnings, duplicate_errors = load_config(args.config_path)
    print("=> Duplicate Warnings: ",duplicate_warnings)
    print("=> Duplicate Errors: ",duplicate_errors)

    args2config(config, args)
    config = munchify(config)
    process_configs(config)
    config.model.model_level = 'node'

    uq_args = dUQArgs(explicit_bool=True).parse_args(known_only=True)
    config.uq = munchify(uq_args)
    anchor_type= config.uq.anchor_type 
    print("=> dUQ Args: ")
    print("=> Anchor Type: ",config.uq.anchor_type) 
    print("=> Num Anchors: ",config.uq.num_anchors) 
    
    """
    Create Loader
    """
    dataset = load_dataset(config.dataset.dataset_name, config)
    loader = create_dataloader(dataset, config)
    train_loader = loader['train']
    id_val = loader['id_val']
    sample = next(iter(train_loader))
    config.dataset.dim_node = sample.x.shape[1] * 2  #intentional [x-c,c]
    
    """
    Create Anchor Distribution
    """
    means = []
    for batch in train_loader:
        means.append(batch.x[batch['train_mask']]) 
    means = torch.cat(means) 
    mu = means.mean(dim=0) 
    std = means.std(dim=0) 
    std[std == 0] = 1e-3 
  
    if config.dataset.num_classes == 1:
        for k,v in dataset.items():
            print(k,v)
            try:
                v.y = torch.nn.functional.one_hot(v.y.long(), num_classes=2).long()
            except:
                pass

    if dataset[0].x.shape[1] == 1 and uq_args.gduq_type == 'input':
        print("=> Please use positional encodings or an input embedding first.") 
        exit()

    """
    Create Model.
    """
    gcn_enc = GCNEncoder(config)
    
    base_net= baseModelNode(encoder=gcn_enc,
        num_classes =config.dataset.num_classes)
    model = GraphANTNode(base_network=base_net,
        mean=mu,
        std =std,
        anchor_type=anchor_type,
        num_classes=config.dataset.num_classes)
    model.to(DEVICE)
    model.eval()
    anchors = model.get_anchors(sample,num_anchors=config.uq.num_anchors).to(DEVICE) 
    print("=> Anchors Shape: ",anchors.shape) 
    """
    Set-up Model for Training 
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=config.train.mile_stones, gamma=0.1)
    
    """
    Set-up Metrics
    """
    cal_metric_l1 = MulticlassCalibrationError(num_classes=config.dataset.num_classes,n_bins=100, norm='l1')#.to(DEVICE)
    cal_metric_l2 = MulticlassCalibrationError(num_classes=config.dataset.num_classes,n_bins=100, norm='l2')#.to(DEVICE)
    cal_metric_max = MulticlassCalibrationError(num_classes=config.dataset.num_classes,n_bins=100, norm='max')#.to(DEVICE)
    print("NUM CLASSES: ",config.dataset.num_classes)
    acc_metric = MulticlassAccuracy(num_classes = config.dataset.num_classes, average='micro')
        
    train_acc = test_node(model,loader=train_loader,config=config, acc_metric=acc_metric,split='train') 
    print("=> Train Acc: ",train_acc)

 
    """
    Training Loop.
    """
    loss_list = []
    acc_list = []
    cal_err_l2_list = []
    train_acc,val_acc= 0,0
    print("config.model.model_level:",config.model.model_level)
    print('epochs: ',config.train.max_epoch) 
    print("=> Loss Function: ",config.metric.loss_func)
    for epoch in tqdm.tqdm(range(config.train.max_epoch),disable=True):
        loss_avg = 0
        model.train()
        start_time = time.time()
        for batch in train_loader:
            batch = batch.to(config.device)

            optimizer.zero_grad()
            
            mask, targets = nan2zero_get_mask(batch, 'train', config)
            node_norm = batch.node_norm 
            edge_weight = batch.edge_norm 

            model_output = model(batch,n_anchors=config.uq.num_anchors,edge_weight=edge_weight) #num_anchors
            loss = config.metric.loss_func(model_output, targets, reduction='none') * mask
            loss = loss * node_norm * mask.sum() 
            loss = loss.sum() / mask.sum()
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_avg += loss
        loss_avg /= len(train_loader) 
        stat_dict = get_results_summary_node(model, 
            loader=id_val,
            anchors=anchors,
            n_anchors=config.uq.num_anchors,
            cal_metric_l1=cal_metric_l1,
            cal_metric_l2=cal_metric_l2,
            cal_metric_max=cal_metric_max,
            acc_metric=acc_metric,
            split='id_val')
        val_acc = stat_dict['acc']
        cal_err_l2 = stat_dict['cal_err_l2'] 
        
        acc_list.append(val_acc.cpu())
        cal_err_l2_list.append(cal_err_l2.cpu())
        loss_list.append(loss_avg.cpu().detach().numpy())
        end_time = time.time() - start_time

        
        train_acc = test_node(model,
            loader=loader['eval_train'],
            config=config, 
            acc_metric=acc_metric,
            split='eval_train') 
        val_acc = test_node(model,
            loader=loader['id_val'],
            config=config, 
            acc_metric=acc_metric,
            split='id_val') 
        print("({5:.3f}) Epoch: {0} -- Loss: {1:.3f} -- Train Acc: {4:.4f} -- Val Acc: {2:.4f} -- CalErrL2: {3:4f}".format(epoch,loss_avg,val_acc,cal_err_l2,train_acc,end_time),flush=True)
     
    for key in sorted(loader.keys()):
        tmp_loader = loader[key] 
        print("====================================")
        print("KEY: ",key)
        if key == 'eval_train':
            pass
        else:
            stat_dict = get_results_summary_node(model, 
                loader=tmp_loader,
                cal_metric_l1=cal_metric_l1,
                cal_metric_l2=cal_metric_l2,
                cal_metric_max=cal_metric_max,
                acc_metric=acc_metric,
                verbose=True,
                anchors=anchors,
                split=key)
            print(f"\tAcc: {stat_dict['acc']:.4f}, CalErrL2: {stat_dict['cal_err_l2']:.4f}")
        print("************************************")
        print()

    """
    Save Model
    """
    ckpt = {
        'encoder':model.net.encoder.state_dict(),
        'classifer':model.net.classifier.state_dict(),
    }
    save_name = "_".join(
        [
            "gduq",
            config["dataset"]["dataset_name"],
            config["dataset"]["domain"],
            config["dataset"]["shift_type"],
            config["model"]["model_name"],
            f"{uq_args.anchor_type}-{uq_args.num_anchors}",
            str(uq_args.layerwise_duq),
            uq_args.gduq_type, 
            str(config["random_seed"]),
        ]
    )

    prefix = f"{uq_args.save_path}/{config.dataset.dataset_name}"
    if os.path.exists(prefix) is False:
        os.makedirs(prefix)
    print("=> Save Name: ",save_name)
    torch.save(ckpt, "{}/{}.ckpt".format(prefix, save_name))

if __name__ == "__main__":
    main()
    print("=> Done!")