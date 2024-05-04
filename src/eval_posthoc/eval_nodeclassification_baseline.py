import numpy as np
import argparse
from munch import munchify
import torch
import torch.nn.functional as F
import tqdm
import random

import torch_geometric as geom
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union
from tap import Tap
from pprint import pprint
from torch_geometric.loader import DataLoader 

sns.set_style("whitegrid")

import GOOD
from GOOD.utils.args import args_parser
from GOOD.utils.config_reader import load_config, args2config, process_configs
from GOOD.data import load_dataset, create_dataloader

from torchmetrics.classification import BinaryCalibrationError, MulticlassCalibrationError,BinaryAccuracy, MulticlassAccuracy

import sys
sys.path.append("/p/lustre3/trivedi1/GDUQ/src")
from GOOD.utils.train import nan2zero_get_mask
from calibration.calibrator_node import ETS,TS,VS,IRM, Dirichlet, OrderInvariantCalib, SplineCalib, CaGCN, GATS 
from calibration.calib_utils import load_node_to_nearest_training, dotdict, get_net_results
from utils import DEVICE, dUQArgs, get_results_summary_node, test_node, count_parameters
from models.encoders import GCNEncoder
from models.gduq_models import baseModelNode, GraphANTNode 

class EvalArgs(Tap):
    uq_name: str = None
    ckpt_path: str = None
    ckpt_list: List[str] = None  # DEns
    num_anchors: int = 10  # duq
    num_samples: int = 100  # mcd

def main():
    print("DEVICE: ", DEVICE)

    args = args_parser()
    config, duplicate_warnings, duplicate_errors = load_config(args.config_path)
    print("=> Duplicate Warnings: ", duplicate_warnings)
    print("=> Duplicate Errors: ", duplicate_errors)

    args2config(config, args)
    config = munchify(config)
    process_configs(config)
    config.model.model_level = 'node'

    uq_args = EvalArgs(explicit_bool=True).parse_args(known_only=True)
    config.uq = munchify(uq_args)
    print("=> Eval Args: ")
    pprint(repr(uq_args))

    save_name = "_".join([
        config.uq.uq_name, 
        config.dataset.dataset_name,
        config.dataset.domain, 
        config.dataset.shift_type,
        config.model.model_name, 
        str(config.random_seed),
        ])
    
    print("=> Save Name: ",save_name)
    """
    Get Dataset.
    Note: Shuffle the loaders to ensure that samples 
    are not sorted by class, if using DUQ 
    """

    dataset = load_dataset(config.dataset.dataset_name, config)
    loader = create_dataloader(dataset, config)
    train_loader = loader['train']
    id_val = loader['id_val']
    sample = next(iter(train_loader))
    config.dataset.dim_node = sample.x.shape[1]  #intentional [x-c,c]
    print("=> NODE DIM: ",sample.x.shape[1])
    print("=> Loaders: ",sorted(loader.keys()))#['train', 'eval_train', 'id_val', 'id_test', 'val', 'test']
    
    """
    Set-up Model.
    """
    gin_enc= GCNEncoder(config)
    
    base_net = baseModelNode(encoder=gin_enc,num_classes =config.dataset.num_classes)
    base_net.to(DEVICE)

    if config.uq.ckpt_path is None:
        ckpt_path = config.uq.ckpt_list[0]
    else:
        ckpt_path = config.uq.ckpt_path
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    base_net.encoder.load_state_dict(ckpt["encoder"])
    base_net.classifier.load_state_dict(ckpt["classifer"])
    base_net.eval()
    print("=> NUM PARAMS: ", count_parameters(base_net)) 
    """
    Set-up Metrics
    """
    if config.dataset.num_classes == 2:
        cal_metric_l1 = BinaryCalibrationError(n_bins=100, norm='l1')
        acc_metric = BinaryAccuracy(multidim_average='global')
    else:
        cal_metric_l1 = MulticlassCalibrationError(num_classes=config.dataset.num_classes,n_bins=100, norm='l1')#.to(DEVICE)
        cal_metric_l2 = MulticlassCalibrationError(num_classes=config.dataset.num_classes,n_bins=100, norm='l2')#.to(DEVICE)
        cal_metric_max = MulticlassCalibrationError(num_classes=config.dataset.num_classes,n_bins=100, norm='max')#.to(DEVICE)
        acc_metric = MulticlassAccuracy(num_classes = config.dataset.num_classes,average='micro')


    # pdb.set_trace()
    t = torch.Tensor([1])
    if config.uq.uq_name.upper() in ['TS', 'VS', 'ETS']:
        cal_wdecay = 0
    elif config.uq.uq_name.upper() == 'CaGCN':
        if config.dataset.dataset_name == "CoraFull":
            cal_wdecay = 0.03
        else:
            cal_wdecay = 5e-3
    else:
        cal_wdecay = 5e-4
    
    if config.uq.uq_name.lower() == "ets":
        temp_model = ETS(base_net, config.dataset.num_classes)
        temp_model.fit(loader['eval_train'][0].to(DEVICE), loader['eval_train'][0]['id_val_mask'],  loader['eval_train'][0]['train_mask'], cal_wdecay)
    if config.uq.uq_name.lower() == "vanilla":
        temp_model = Vanilla(base_net, config.dataset.num_classes)
        temp_model.fit(loader['eval_train'][0].to(DEVICE), loader['eval_train'][0]['id_val_mask'],  loader['eval_train'][0]['train_mask'], cal_wdecay)
    elif config.uq.uq_name.lower() == "ts":
        temp_model = TS(base_net)
        temp_model.fit(loader['eval_train'][0].to(DEVICE), loader['eval_train'][0]['id_val_mask'],  loader['eval_train'][0]['train_mask'], cal_wdecay)

        # temp_model.fit(loader['id_val'], loader['eval_train'], cal_wdecay)
    elif config.uq.uq_name.lower() == "vs":
        temp_model = VS(base_net,config.dataset.num_classes)
        temp_model.fit(loader['eval_train'][0].to(DEVICE), loader['eval_train'][0]['id_val_mask'],  loader['eval_train'][0]['train_mask'], cal_wdecay)
    elif config.uq.uq_name.lower() == "irm":
        temp_model = IRM(base_net)
        temp_model.fit(loader['eval_train'][0].to(DEVICE), loader['eval_train'][0]['id_val_mask'],  loader['eval_train'][0]['train_mask'], cal_wdecay) 
    elif config.uq.uq_name.lower() == "dirichlet":
        temp_model = Dirichlet(base_net,config.dataset.num_classes)
        temp_model.fit(loader['eval_train'][0].to(DEVICE), loader['eval_train'][0]['id_val_mask'],  loader['eval_train'][0]['train_mask'], cal_wdecay) 
    elif config.uq.uq_name.lower() == "spline":
        temp_model = SplineCalib(base_net,7)
        temp_model.fit(loader['eval_train'][0].to(DEVICE), loader['eval_train'][0]['id_val_mask'],  loader['eval_train'][0]['train_mask'], cal_wdecay) 
    elif config.uq.uq_name.lower() == "orderinvariant":
        temp_model = OrderInvariantCalib(base_net,config.dataset.num_classes)
        temp_model.fit(loader['eval_train'][0].to(DEVICE), loader['eval_train'][0]['id_val_mask'],  loader['eval_train'][0]['train_mask'], cal_wdecay) 
    elif config.uq.uq_name.lower() == 'cagcn':
        temp_model = CaGCN(base_net, loader['eval_train'][0].num_nodes, config.dataset.num_classes, 0.5)
        temp_model.fit(loader['eval_train'][0].to(DEVICE), loader['eval_train'][0]['id_val_mask'],  loader['eval_train'][0]['train_mask'], cal_wdecay) 
    elif config.uq.uq_name.lower() == 'gats':
        """
        dot dict 
        """
        gats_args = {}
        gats_args['cal_wdecay'] = 0
        gats_args['heads'] = 8
        gats_args['bias'] = 1

        gats_args =  dotdict(gats_args)
        data = loader['eval_train'][0]
        dist_to_train = load_node_to_nearest_training(config.dataset.dataset_name, loader['eval_train'][0])
        temp_model = GATS(base_net, data.edge_index.to(DEVICE), data.num_nodes, data['train_mask'].to(DEVICE),
                        dataset.num_classes, dist_to_train, gats_args) 

        temp_model.fit(loader['eval_train'][0].to(DEVICE), loader['eval_train'][0]['id_val_mask'].to(DEVICE),  loader['eval_train'][0]['train_mask'].to(DEVICE), cal_wdecay) 
    else:
        t = torch.Tensor([1])
        print("=> Default Temperature: ", np.round(t, 4)) 
    
    
    EVAL_KEYS= ['eval_train','id_test','id_val','test','val'] 
    """
    Vanilla 
    """
    acc_list = []
    
    LOG_PATH = "logs.csv"
    
    print("*" * 50)
    for loader_key in sorted(EVAL_KEYS):
        # print("\t{0}".format(loader_key))
        logits, conf, correct, l = get_net_results(
            temp_model, 
            loader[loader_key],
            config=config, 
            in_dist=True, 
            t=1, 
            device=DEVICE,
            split=loader_key
        )
        acc_list.append(test_node(model=base_net, 
            loader=loader[loader_key], 
            acc_metric=acc_metric,
            config=config,
            anchors=None,
            split=loader_key))
        logits = torch.Tensor(np.array(logits))
        labels = torch.LongTensor(l)
    
        cal_err_l1 = cal_metric_l1(logits,labels)
        cal_err_l2 = cal_metric_l2(logits,labels)
        cal_err_max = cal_metric_max(logits,labels)
        acc = acc_metric(logits,labels)
        acc = str(np.round(acc.item(),4))
        cal_err_l1 = str(np.round(cal_err_l1.item(),4))
        cal_err_l2 = str(np.round(cal_err_l2.item(),4))
        cal_err_max = str(np.round(cal_err_max.item(),4))
            
        save_str = ",".join([save_name,
            loader_key,
            acc, 
            cal_err_l1,
            cal_err_l2,
            cal_err_max,
            ])
        print("=> {}".format(save_str))
        with open(LOG_PATH,'a') as f:
            f.write("{}\n".format(save_str))
    print("*" * 50)




if __name__ == "__main__":
    main()