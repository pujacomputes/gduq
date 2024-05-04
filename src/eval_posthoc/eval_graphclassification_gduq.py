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


from torchmetrics.classification import BinaryCalibrationError, MulticlassCalibrationError,BinaryAccuracy, MulticlassAccuracy,BinaryAUROC

import sys
sys.path.append("/p/lustre3/trivedi1/GDUQ/src")

from calibration.calib_utils import get_results_summary
from utils import DEVICE,dUQArgs

from pytorch_ood.detector import MaxSoftmax
from pytorch_ood.utils import OODMetrics, ToUnknown
from torch_geometric.loader import DataLoader
from calibration.graph_calibrator_duq import ETS,TS,VS,IRM, Dirichlet, OrderInvariantCalib, SplineCalib

from models.base_models import baseModel, baseModelvGIN
from models.encoders import GINEncoder, GINEncoderLayerwise, vGINEncoder, vGINEncoderLayerwise 
from models.gduq_models import GraphANT,GraphANTHiddenReps, GraphANTLayerwiseGIN, GraphANTLayerwisevGIN 

class MaxSigmoid():
    def __init__(self,net) -> None:
        self.net = net
    def score(self,raw_logits):
        sig_logits = torch.nn.functional.sigmoid(raw_logits)
        sig_conf = torch.maximum(sig_logits, 1-sig_logits)
        return sig_conf 


class ToUnknown(object):
    def __call__(self, data):
        #this is done for the OOD dataset.
        desired_shape = data.y.shape
        data.y = torch.LongTensor([-1]).reshape(desired_shape) 
        return data

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
    config.model.model_level = 'graph'

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
    num_feats = dataset['train'][0].x.shape[1]
    print("=> Num Feats: ",num_feats)


    """
    Create Anchor Distribution
    """
    config.dataset.dim_node = dataset['train'][0].x.shape[1] 
    anchor_type = config.uq.anchor_type
    gduq_type = config.uq.gduq_type
    """
    Create Model.
    """

    ckpt = torch.load(config.uq.ckpt_path)
    try:
        mean = ckpt['mu']
        std = ckpt['std']
    except:
        mean = None 
        std = None

    """
    Create Model.
    """
    if uq_args.gduq_type == 'layerwise':
        if config.model.model_name == "vGIN": 
            print("=> Using vGIN")
            gin_enc = vGINEncoderLayerwise(config)
            base_net = baseModel(encoder=gin_enc, num_classes=config.dataset.num_classes)
        else:
            gin_enc = GINEncoderLayerwise(config)
            print("=> Using GIN")
            base_net = baseModelvGIN(encoder=gin_enc, num_classes=config.dataset.num_classes)
        model = GraphANTLayerwisevGIN(base_network=base_net,
            mean=None,
            std=None,
            anchor_type=anchor_type,
            num_classes=config.dataset.num_classes) 
        
    elif uq_args.gduq_type == 'hidden':
        uq_args.layerwise_duq = -1
        if config.model.model_name == "vGIN": 
            gin_enc = vGINEncoder(config)
        else:
            gin_enc = GINEncoder(config)
        base_net = baseModel(encoder=gin_enc,
            num_classes =config.dataset.num_classes,
            classifier_dim=600)
        model = GraphANTHiddenReps(base_network=base_net,
            mean=None,
            std =None,
            anchor_type=anchor_type,
            num_classes=config.dataset.num_classes)

    elif uq_args.gduq_type == 'input':
        uq_args.layerwise_duq = -1
        x = torch.zeros(len(dataset["train"]), dataset["train"][0].x.shape[1])
        for enum, d in enumerate(dataset["train"]):
            x[enum] = d.x.mean(dim=0)
        x = (x - x.min(dim=0)[0]) / (x.max(dim=0)[0] - x.min(dim=0)[0])
        mu = x.mean(dim=0)
        std = x.std(dim=0)
        if std.norm() < 1e-2:  # if the std is very small, update it.
            std = 0.5
            print("=> Updating Std: ", std)
        config.dataset.dim_node = dataset["train"][0].x.shape[1] * 2  # intentional [x-c,c]
        if config.model.model_name == "vGIN":
            print("=> Using vGIN")
            gin_enc = vGINEncoder(config)
        else:
            gin_enc = GINEncoder(config)
            print("=> Using GIN")

        base_net = baseModel(encoder=gin_enc, num_classes=config.dataset.num_classes)
        model = GraphANT(
            base_network=base_net,
            mean=mu,
            std=std,
            anchor_type=anchor_type,
            num_classes=config.dataset.num_classes,
        )

    else:
        print("=> WARNING WARNING WARNING. INVALID ANCHORING STRATEGY. EXITING!!")
        exit()
  
    model.to(DEVICE)
    print("=> Anchor Type: ",config.uq.anchor_type)

    ckpt = torch.load(config.uq.ckpt_path)
    model.net.encoder.load_state_dict(ckpt['encoder'])
    model.net.classifier.load_state_dict(ckpt['classifer'])
    model.eval()
    print("=> MODEL IS LOADED")
    
    '''
    Load Data 
    '''
    inputs = next(iter(id_val)) 
    inputs = inputs.to(DEVICE)
    print("=> Input Shape: ",inputs.x.shape)
    print("=> Anchor Type: ",anchor_type)

    anchors = model.get_anchors(inputs,num_anchors=config.uq.num_anchors).to(DEVICE) 
    print("=> Input Shape: ",inputs.x.shape)
    print("=> Anchors Shape: ",anchors.shape)
    
    """
    Set-up Metrics
    """
    if config.dataset.num_classes <= 2:
        cal_metric_l1 = BinaryCalibrationError(n_bins=100, norm="l1")
        cal_metric_l2 = BinaryCalibrationError(n_bins=100, norm="l2")  # .to(DEVICE)
        cal_metric_max = BinaryCalibrationError(n_bins=100, norm="max")  # .to(DEVICE)
        acc_metric = BinaryAccuracy(multidim_average='global')
        gengap_acc_metric = BinaryAccuracy(multidim_average='global')
        use_binary=True
    else:
        cal_metric_l1 = MulticlassCalibrationError(num_classes=config.dataset.num_classes,n_bins=100, norm='l1')#.to(DEVICE)
        cal_metric_l2 = MulticlassCalibrationError(num_classes=config.dataset.num_classes,n_bins=100, norm='l2')#.to(DEVICE)
        cal_metric_max = MulticlassCalibrationError(num_classes=config.dataset.num_classes,n_bins=100, norm='max')#.to(DEVICE)
        acc_metric = MulticlassAccuracy(num_classes = config.dataset.num_classes,average='micro')
        gengap_acc_metric = MulticlassAccuracy(num_classes = config.dataset.num_classes,average='micro')

    if "pretrain" in config.uq.ckpt_path:
        save_name = "_".join(["pretrain-gduqhiddenrep",
            config.uq.uq_name.upper(),  
            config['dataset']['dataset_name'],
            config['dataset']['domain'],
            config['dataset']['shift_type'],
            config['model']['model_name'],
            "{}-{}".format(anchor_type,config.uq.num_anchors),
            str(config['random_seed'])])
    else:
        save_name = "_".join(["gduqhiddenrep",
            config.uq.uq_name.upper(), 
            config['dataset']['dataset_name'],
            config['dataset']['domain'],
            config['dataset']['shift_type'],
            config['model']['model_name'],
            "{}-{}".format(anchor_type,config.uq.num_anchors),
            str(config['random_seed'])])
    print("=> Save Name: ",save_name)
    
    LOG_PATH = "logs.csv"
    """
    Getting calibration
    """

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
        print("========> USING ETS")
        temp_model = ETS(model, config.dataset.num_classes)
        temp_model.anchors = anchors
        temp_model.n_anchors = config.uq.num_anchors 
        temp_model.fit(loader['id_val'], loader['eval_train'], cal_wdecay)
    elif config.uq.uq_name.lower() == "ts":
        print("========> USING TS")
        temp_model = TS(model)
        temp_model.anchors = anchors
        temp_model.n_anchors = config.uq.num_anchors 
        temp_model.fit(loader['id_val'], loader['eval_train'], cal_wdecay)
    elif config.uq.uq_name.lower() == "vs":
        print("========> USING VS")
        temp_model = VS(model,config.dataset.num_classes)
        temp_model.anchors = anchors
        temp_model.n_anchors = config.uq.num_anchors 
        temp_model.fit(loader['id_val'], loader['eval_train'], cal_wdecay)
    elif config.uq.uq_name.lower() == "irm":
        print("========> USING IRM")
        temp_model = IRM(model)
        temp_model.anchors = anchors
        temp_model.n_anchors = config.uq.num_anchors 
        temp_model.fit(loader['id_val'], loader['eval_train'], cal_wdecay)
    elif config.uq.uq_name.lower() == "dirichlet":
        print("========> USING DIRICHLET")
        temp_model = Dirichlet(model,config.dataset.num_classes)
        temp_model.anchors = anchors
        temp_model.n_anchors = config.uq.num_anchors 
        temp_model.fit(loader['id_val'], loader['eval_train'], cal_wdecay)
    elif config.uq.uq_name.lower() == "spline":
        print("========> USING SPLINE")
        temp_model = SplineCalib(model,7)
        temp_model.anchors = anchors
        temp_model.n_anchors = config.uq.num_anchors 
        temp_model.fit(loader['id_val'], loader['eval_train'], cal_wdecay)
    elif config.uq.uq_name.lower() == "orderinvariant":
        print("========> USING ORDERINVARIANT")
        temp_model = OrderInvariantCalib(model,config.dataset.num_classes)
        temp_model.anchors = anchors
        temp_model.n_anchors = config.uq.num_anchors 
        temp_model.fit(loader['id_val'], loader['eval_train'], cal_wdecay)
    else:
        t = torch.Tensor([1])
        temp_model = TS(model)
        temp_model.anchors = anchors
        temp_model.n_anchors = config.uq.num_anchors
        temp_model = temp_model.to(DEVICE) 
        print("=> Default Temperature: ", np.round(t, 4))



    loader.pop("eval_train")
    print("==============CALIBRATION======================")
    for key in sorted(loader.keys()):
            tmp_loader = loader[key] 
            print("====================================")
            print("KEY: ",key)
            #Have already assigned the anchor and n_anchors to temp model
            #It gets th calibrated logits, and scales are needed.
            stat_dict = get_results_summary(temp_model, 
                anchors=None, 
                loader=tmp_loader,
                n_anchors=None,
                cal_metric_l1=cal_metric_l1,
                cal_metric_l2=cal_metric_l2,
                cal_metric_max=cal_metric_max,
                acc_metric=acc_metric,
                )
            acc = str(np.round(stat_dict['acc'].item(),4))
            cal_err_l1 = str(np.round(stat_dict['cal_err_l1'].item(),4))
            cal_err_l2 = str(np.round(stat_dict['cal_err_l2'].item(),4))
            cal_err_max = str(np.round(stat_dict['cal_err_max'].item(),4))
            save_str = ",".join([save_name,key,acc, cal_err_l1,cal_err_l2,cal_err_max])
            print("=> {}".format(save_str))
            with open(LOG_PATH,'a') as f:
                f.write("{}\n".format(save_str))
    print("************************************")
    print()      
