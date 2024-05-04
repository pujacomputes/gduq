import numpy as np
import pdb
import tqdm
import time

from tap import Tap
from pprint import pprint
from munch import Munch 
from munch import munchify

import torch.nn as nn
import torchvision.transforms as transforms
import torch
import torchvision.models as models

from GOOD.data import load_dataset, create_dataloader
from GOOD.utils.args import args_parser
from GOOD.utils.config_reader import load_config, args2config, process_configs

from torchmetrics.classification import BinaryCalibrationError, MulticlassCalibrationError,BinaryAccuracy, MulticlassAccuracy
from torchmetrics.functional.classification import multiclass_accuracy

# from calibration_tools import calib_err,get_measures, show_calibration_results
from GOOD.utils.train import nan2zero_get_mask

import sys
sys.path.append("/p/lustre3/trivedi1/GDUQ/src")

from calibration.calibrator_duq import ETS,TS,VS,IRM, Dirichlet, OrderInvariantCalib, SplineCalib, CaGCN, GATS
from calibration.calib_utils import load_node_to_nearest_training, dotdict, get_net_results 
from utils import DEVICE, dUQArgs, get_results_summary_node, test_node, count_parameters
from models.encoders import GCNEncoder
from models.gduq_models import baseModelNode, GraphANTNode 


if __name__=='__main__':
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
    print("=> dUQ Args: ")
    print("=> Anchor Type: ",config.uq.anchor_type) 
    print("=> Num Anchors: ",config.uq.num_anchors) 
    print("=> Ckpt Path: ",config.uq.ckpt_path) 

    '''
    Load Data 
    '''
    dataset = load_dataset(config.dataset.dataset_name, config)
    loader = create_dataloader(dataset, config)
    train_loader = loader['train']
    id_val = loader['id_val']
    inputs = next(iter(train_loader))
    config.dataset.dim_node = inputs.x.shape[1] * 2 #intentional [x-c,c]
    print("=> Num Feats: ",config.dataset.dim_node)
    
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
    anchor_type= config.uq.anchor_type

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

    gin_enc = GCNEncoder(config)
    base_net = baseModelNode(encoder=gin_enc,num_classes =config.dataset.num_classes)
    model = GraphANTNode(base_network=base_net,
        mean=mu,
        std =std,
        anchor_type=anchor_type,
        num_classes=config.dataset.num_classes)
    model.to(DEVICE)
    model.eval()
    print("=> Anchor Type: ",anchor_type)

    ckpt = torch.load(config.uq.ckpt_path)
    model.net.encoder.load_state_dict(ckpt['encoder'])
    model.net.classifier.load_state_dict(ckpt['classifer'])
    print("=> MODEL IS LOADED")
    
    '''
    Load Data 
    '''

    if anchor_type == 'node2node':
        anchors=None
        print("DIFFERENT ANCHOR PER NODE!!")
    else:
        anchors = model.get_anchors(inputs,num_anchors=config.uq.num_anchors).to(DEVICE) 
        print("=> Anchors Shape: ",anchors.shape)
    print("=> Input Shape: ",inputs.x.shape)
    
    """
    Set-up Metrics
    """
    if config.dataset.num_classes < 2:
        cal_metric_l1 = BinaryCalibrationError(n_bins=100, norm="l1")
        cal_metric_l2 = BinaryCalibrationError(n_bins=100, norm="l2")  # .to(DEVICE)
        cal_metric_max = BinaryCalibrationError(n_bins=100, norm="max")  # .to(DEVICE)
        acc_metric = BinaryAccuracy(multidim_average='global')
    else:
        cal_metric_l1 = MulticlassCalibrationError(num_classes=config.dataset.num_classes,n_bins=100, norm='l1')#.to(DEVICE)
        cal_metric_l2 = MulticlassCalibrationError(num_classes=config.dataset.num_classes,n_bins=100, norm='l2')#.to(DEVICE)
        cal_metric_max = MulticlassCalibrationError(num_classes=config.dataset.num_classes,n_bins=100, norm='max')#.to(DEVICE)
        acc_metric = MulticlassAccuracy(num_classes = config.dataset.num_classes,average='micro')

    save_name = "_".join(["gduqfeature",
        config.uq.uq_name.upper(), 
        config['dataset']['dataset_name'],
        config['dataset']['domain'],
        config['dataset']['shift_type'],
        config['model']['model_name'],
        "{}-{}".format(anchor_type,config.uq.num_anchors),
        str(config['random_seed'])])
    print("=> Save Name: ",save_name)
    print("=> NUM PARAMS BaseNet: ",count_parameters(base_net))
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
        temp_model = ETS(model, config.dataset.num_classes)
        temp_model.anchors = anchors
        temp_model.n_anchors = config.uq.num_anchors 
        temp_model.fit(loader['eval_train'][0].to(DEVICE), loader['eval_train'][0]['id_val_mask'],  loader['eval_train'][0]['train_mask'], cal_wdecay)
    elif config.uq.uq_name.lower() == "ts":
        temp_model = TS(model)
        temp_model.anchors = anchors
        temp_model.n_anchors = config.uq.num_anchors 
        temp_model.fit(loader['eval_train'][0].to(DEVICE), loader['eval_train'][0]['id_val_mask'],  loader['eval_train'][0]['train_mask'], cal_wdecay)
    elif config.uq.uq_name.lower() == "vs":
        temp_model = VS(model,config.dataset.num_classes)
        temp_model.anchors = anchors
        temp_model.n_anchors = config.uq.num_anchors 
        temp_model.fit(loader['eval_train'][0].to(DEVICE), loader['eval_train'][0]['id_val_mask'],  loader['eval_train'][0]['train_mask'], cal_wdecay)
    elif config.uq.uq_name.lower() == "irm":
        temp_model = IRM(model)
        temp_model.anchors = anchors
        temp_model.n_anchors = config.uq.num_anchors 
        temp_model.fit(loader['eval_train'][0].to(DEVICE), loader['eval_train'][0]['id_val_mask'],  loader['eval_train'][0]['train_mask'], cal_wdecay) 
    elif config.uq.uq_name.lower() == "dirichlet":
        temp_model = Dirichlet(model,config.dataset.num_classes)
        temp_model.anchors = anchors
        temp_model.n_anchors = config.uq.num_anchors 
        temp_model.fit(loader['eval_train'][0].to(DEVICE), loader['eval_train'][0]['id_val_mask'],  loader['eval_train'][0]['train_mask'], cal_wdecay) 
    elif config.uq.uq_name.lower() == "spline":
        temp_model = SplineCalib(model,7)
        temp_model.anchors = anchors
        temp_model.n_anchors = config.uq.num_anchors 
        temp_model.fit(loader['eval_train'][0].to(DEVICE), loader['eval_train'][0]['id_val_mask'],  loader['eval_train'][0]['train_mask'], cal_wdecay) 
    elif config.uq.uq_name.lower() == "orderinvariant":
        temp_model = OrderInvariantCalib(model,config.dataset.num_classes)
        temp_model.anchors = anchors
        temp_model.n_anchors = config.uq.num_anchors 
        temp_model.fit(loader['eval_train'][0].to(DEVICE), loader['eval_train'][0]['id_val_mask'],  loader['eval_train'][0]['train_mask'], cal_wdecay) 
    elif config.uq.uq_name.lower() == 'cagcn':
        temp_model = CaGCN(model, loader['eval_train'][0].num_nodes, config.dataset.num_classes, 0.5)
        temp_model.anchors = anchors
        temp_model.n_anchors = config.uq.num_anchors 
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
        temp_model = GATS(model, data.edge_index.to(DEVICE), data.num_nodes, data['train_mask'].to(DEVICE),
                        dataset.num_classes, dist_to_train, gats_args) 
        temp_model.anchors = anchors
        temp_model.n_anchors = config.uq.num_anchors 
        temp_model.fit(loader['eval_train'][0].to(DEVICE), loader['eval_train'][0]['id_val_mask'].to(DEVICE),  loader['eval_train'][0]['train_mask'].to(DEVICE), cal_wdecay) 
    else:
        exit()


    acc_list = []
    LOG_PATH = "logs.csv"
    print("====================================")
    EVAL_KEYS= ['eval_train','id_test','id_val','test','val'] 
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
