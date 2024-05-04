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
from GOOD.networks.models.GINs import GINEncoder
from GOOD.networks.models.GINvirtualnode import vGINEncoder
from GOOD.utils.args import args_parser
from GOOD.utils.config_reader import load_config, args2config, process_configs

import sys
sys.path.append("/p/lustre3/trivedi1/GDUQ/src")

from torchmetrics.classification import BinaryCalibrationError, MulticlassCalibrationError,BinaryAccuracy, MulticlassAccuracy,BinaryAUROC
from utils import get_results_summary,DEVICE,dUQArgs,get_logits_labels, count_parameters


from models.gduq_models import GraphANT,GraphANTHiddenReps, GraphANTLayerwiseGIN, GraphANTLayerwisevGIN 
from models.base_models import baseModel, baseModelvGIN, baseModelGINLayerwise, baseModelvGINLayerwise
from models.encoders import GINEncoder, GINEncoderLayerwise, vGINEncoder, vGINEncoderLayerwise 

from pytorch_ood.detector import MaxSoftmax
from pytorch_ood.utils import OODMetrics, ToUnknown
from torch_geometric.loader import DataLoader

class MaxSigmoid():
    def __init__(self,net) -> None:
        self.net = net
    def score(self,raw_logits):
        sig_logits = torch.nn.functional.sigmoid(raw_logits)
        sig_conf = torch.maximum(sig_logits, 1-sig_logits)
        return sig_conf 


def get_logits_labels_disagreement(model,loader,anchors=None,n_anchors=None):
    model.eval()
    preds = []
    targets = []
    unqs=[]
    with torch.no_grad(): 
        for batch in tqdm.tqdm(loader,disable=True):
            p= model.forward_preds(batch.to(DEVICE),
                    anchors=anchors, 
                    n_anchors=n_anchors,
                    return_std=False)
             
            labeled = batch.y == batch.y
            preds.append(p[:,labeled,:])
            targets.append(batch.y[labeled])
    # pdb.set_trace()
    preds = torch.cat(preds,dim=1).to('cpu') 
    targets = torch.cat(targets,dim=0).to('cpu')
    return preds,targets


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
    anchor_type= config.uq.anchor_type

    """
    Create Model.
    """

    if config.model.model_name == "vGIN":
        print("=> Using vGIN")
        gin_enc = vGINEncoder(config)
    else:
        gin_enc = GINEncoder(config)
        print("=> Using GIN")

    ckpt = torch.load(config.uq.ckpt_path)
    try:
        mean = ckpt['mu']
        std = ckpt['std']
    except:
        mean = None 
        std = None
    print(mean,std) 
    anchor_type = config.uq.anchor_type

    base_net = baseModel(encoder=gin_enc,
        classifier_dim=600,
        num_classes =config.dataset.num_classes)
    model = GraphANTHiddenReps(base_network=base_net,
        mean=mean,
        std =std,
        anchor_type=config.uq.anchor_type,
        num_classes=config.dataset.num_classes)
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
        acc_metric = BinaryAccuracy(num_classes = config.dataset.num_classes,average='micro')
        gengap_acc_metric = BinaryAccuracy(num_classes = config.dataset.num_classes,average='micro')
        use_binary=True
        if config.dataset.dataset_name == 'GOODHIV':
            acc_metric = BinaryAUROC(num_classes = config.dataset.num_classes) 
            print("=> USING AUROC INSTEAD!!") 
    else:
        cal_metric_l1 = MulticlassCalibrationError(num_classes=config.dataset.num_classes,n_bins=100, norm='l1')#.to(DEVICE)
        cal_metric_l2 = MulticlassCalibrationError(num_classes=config.dataset.num_classes,n_bins=100, norm='l2')#.to(DEVICE)
        cal_metric_max = MulticlassCalibrationError(num_classes=config.dataset.num_classes,n_bins=100, norm='max')#.to(DEVICE)
        acc_metric = MulticlassAccuracy(num_classes = config.dataset.num_classes,average='micro')
        gengap_acc_metric = MulticlassAccuracy(num_classes = config.dataset.num_classes,average='micro')

    if "pretrain" in config.uq.ckpt_path:
        save_name = "_".join(["pretrain-gduqhiddenrep", 
            config['dataset']['dataset_name'],
            config['dataset']['domain'],
            config['dataset']['shift_type'],
            config['model']['model_name'],
            "{}-{}".format(anchor_type,config.uq.num_anchors),
            str(config['random_seed'])])
    else:
        save_name = "_".join(["gduqhiddenrep", 
            config['dataset']['dataset_name'],
            config['dataset']['domain'],
            config['dataset']['shift_type'],
            config['model']['model_name'],
            "{}-{}".format(anchor_type,config.uq.num_anchors),
            str(config['random_seed'])])
    print("=> Save Name: ",save_name)
    print("=> NUM PARAMS: ",count_parameters(base_net), count_parameters(model))
    exit() 
    LOG_PATH = "/usr/workspace/trivedi1/Fall2022/iclr22-graphduq-experiments/extended-gduq-eval/hiddenrep-anchor_cal_logs.csv"
    
    """
    Getting calibration
    """
    print("==============CALIBRATION======================")
    for key in sorted(loader.keys()):
            tmp_loader = loader[key] 
            print("====================================")
            print("KEY: ",key)
            stat_dict = get_results_summary(model, 
                anchors=anchors, 
                loader=tmp_loader,
                n_anchors=config.uq.num_anchors,
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

    """
    Generalization Gap Prediction 
    Predict ID and OOD test and accuracy 
    """

    GENGAP_PATH= "/usr/workspace/trivedi1/Fall2022/iclr22-graphduq-experiments/extended-gduq-eval/hiddenrep-anchor_gengap_logs.csv"
    id_probs, id_labels, id_unqs = get_logits_labels(model=model, 
                                                        loader=loader['id_val'], 
                                                        anchors=anchors,
                                                        n_anchors=config.uq.num_anchors)
    if config.dataset.num_classes > 2: 
        id_probs = torch.nn.functional.softmax(id_probs,dim=1)
        id_confs = id_probs.max(dim=1)[0]
        id_preds = id_probs.argmax(dim=1)
    else: 
        id_probs = torch.nn.functional.sigmoid(id_probs)
        id_confs = torch.maximum(id_probs, 1-id_probs)
        id_preds = (id_probs > 0.5).float()
    id_val_acc = gengap_acc_metric(id_preds,id_labels).item() 

    target_probs, target_labels, target_unqs = get_logits_labels(model=model, 
                                                    loader=loader['test'], 
                                                    anchors=anchors,
                                                    n_anchors=config.uq.num_anchors) 
    if config.dataset.num_classes > 2: 
        target_probs = torch.nn.functional.softmax(target_probs,dim=1)
        scores = target_confs = target_probs.max(dim=1)[0]
        target_preds = target_probs.argmax(dim=1)
    else:
        target_probs = torch.nn.functional.sigmoid(target_probs)
        scores = target_confs = torch.maximum(target_probs, 1-target_probs)
        target_preds = (target_probs > 0.5).float()
    ood_test_acc = gengap_acc_metric(target_preds,target_labels).item()

    ood_val_probs, ood_val_labels, ood_val_unqs = get_logits_labels(model=model, 
                                                    loader=loader['val'], 
                                                    anchors=anchors,
                                                    n_anchors=config.uq.num_anchors) 
    if config.dataset.num_classes > 2: 
        ood_val_probs = torch.nn.functional.softmax(ood_val_probs,dim=1)
        ood_val_confs = ood_val_probs.max(dim=1)[0]
        ood_val_preds = ood_val_probs.argmax(dim=1)
    else: 
        ood_val_probs = torch.nn.functional.sigmoid(ood_val_probs)
        ood_val_confs = torch.maximum(ood_val_probs, 1-ood_val_probs)
        ood_val_preds = (ood_val_probs > 0.5).float()

    ood_val_acc = gengap_acc_metric(ood_val_preds,ood_val_labels).item()
    
    
    id_test_probs,id_test_labels, id_test_unqs = get_logits_labels(model=model, 
                                                    loader=loader['id_test'], 
                                                    anchors=anchors,
                                                    n_anchors=config.uq.num_anchors) 
    
    if config.dataset.num_classes > 2: 
        id_test_probs = torch.nn.functional.softmax(id_test_probs,dim=1)
        id_test_confs = id_test_probs.max(dim=1)[0]
        id_test_preds = id_test_probs.argmax(dim=1)
    else: 
        id_test_probs = torch.nn.functional.sigmoid(id_test_probs)
        id_test_preds = (id_test_probs > 0.5).float()
        id_test_confs = torch.maximum(id_test_probs, 1-id_test_probs)
    
    id_test_acc = gengap_acc_metric(id_test_preds,id_test_labels).item() 
    
    """
    Compute ID Val threshold
    """
    thresholds = np.arange(0, 1, 0.001) 
    id_acc_at_thres = []
    ood_acc_at_thres = []
    for thres in thresholds:
        id_acc_at_thres.append((id_confs >= thres).sum() / len(id_confs))
        ood_acc_at_thres.append((ood_val_confs >= thres).sum() / len(ood_val_confs))
    id_val_idx = np.abs(np.array(id_acc_at_thres) - id_val_acc).argmin()
    ood_val_idx = np.abs(np.array(ood_acc_at_thres) - ood_val_acc).argmin()
    # pdb.set_trace() 
    id_val_found_thres = thresholds[id_val_idx]
    ood_val_found_thres = thresholds[ood_val_idx]
    
    pred_target_acc_w_id_thres = (target_confs >= id_val_found_thres).sum() / len(target_confs)
    pred_target_acc_w_ood_thres = (target_confs >= ood_val_found_thres).sum() / len(target_confs)
    pred_id_test_acc_w_id_thres = (id_test_confs >= id_val_found_thres).sum() / len(id_test_confs)
    
    mae_diff_w_id_thres = np.abs(ood_test_acc- pred_target_acc_w_id_thres)
    mae_diff_id_test_w_id_thres = np.abs(id_test_acc - pred_id_test_acc_w_id_thres)
    mae_diff_w_ood_thres = np.abs(ood_test_acc- pred_target_acc_w_ood_thres)


    save_str_id= "{id_thres:.4f},{id_pred_acc:.4f},{id_true_acc:.4f},{id_mae:.4f},{ood_pred:.4f}, {ood_true_acc:.4f}, {ood_mae:.4f},{ood_thres:.4f}, {ood_pred_ood_acc:.3f}, {ood_ood_true_acc:.3f}, {ood_ood_mae:.3f} ".format(id_thres=id_val_found_thres,
                                                                                                                                                id_pred_acc = pred_id_test_acc_w_id_thres,
                                                                                                                                                id_true_acc=id_test_acc,
                                                                                                                                                id_mae= mae_diff_id_test_w_id_thres,
                                                                                                                                                ood_pred=pred_target_acc_w_id_thres,
                                                                                                                                                ood_true_acc=ood_test_acc,
                                                                                                                                                ood_mae=mae_diff_w_id_thres,
                                                                                                                                                mae=mae_diff_w_id_thres, 
                                                                                                                                                ood_thres=ood_val_found_thres,
                                                                                                                                                ood_pred_ood_acc = pred_target_acc_w_ood_thres,
                                                                                                                                                ood_ood_true_acc=ood_test_acc,
                                                                                                                                                ood_ood_mae=mae_diff_w_ood_thres 
                                                                                                                                                )

    

    print("=> GENGAP: ",save_str_id)
    with open(GENGAP_PATH,'a') as f:
        f.write("{},{}\n".format(save_name,save_str_id))


    """
    OOD Detection
    (i) scaled calibrated logits.
    (ii) can directly use unqs too.

    IMPORTANT! We will change the labels of the OOD val and OOD test 
    datasets so that we can do OOD detection. 

    So, make sure this is the last evaluation you run. 
    """
    OOD_PATH= "/usr/workspace/trivedi1/Fall2022/iclr22-graphduq-experiments/extended-gduq-eval/hiddenrep-anchor_ood_detection_logs.csv"
    id_val_loader  = loader['id_val'] 
    for key in sorted(['val','test']):
        tmp_loader = loader[key]
        to_u = ToUnknown()
        tmp_dataset = [to_u(d) for d in tmp_loader.dataset]
        tmp_loader = DataLoader(tmp_dataset)

        if config.dataset.num_classes > 2: 
            detector = MaxSoftmax(model)
        else:
            detector = MaxSigmoid(model)
        
        metrics = OODMetrics()
        metrics.reset()


        id_p, id_t,id_unqs = get_logits_labels(model=model, loader=id_val_loader,anchors=anchors,
                                                n_anchors=config.uq.num_anchors)
        ood_p, ood_t,ood_unqs= get_logits_labels(model=model, loader=tmp_loader,anchors=anchors,
                                                n_anchors=config.uq.num_anchors)
            
        if config.dataset.num_classes > 2:
            preds = torch.vstack([id_p,ood_p])
            unqs = torch.vstack([id_unqs,ood_unqs])
        else:
            preds = torch.cat([id_p,ood_p]) 
            unqs = torch.cat([id_unqs,ood_unqs])
        
        targets = torch.cat([id_t,ood_t])

        
        preds = detector.score(preds) 
        metrics.update(preds, targets)
        m_dict_calibrated = metrics.compute()
        # print("=>{}, Calibrated Logits Score: {}".format(key,m_dict_calibrated))
        
        metrics.reset()
        unqs = unqs.mean(dim=1)
        unqs_scaled = (unqs - unqs.min()) / (unqs.max() - unqs.min())
        metrics.update(unqs_scaled, targets)
        m_dict_unqs_scaled = metrics.compute()
        # print("=>{}, UNQS Scaled Score: {}".format(key,m_dict_unqs_scaled))

        save_str = ",".join([save_name,key])
        save_str= "{save_str},{auroc:.4f},{aupr_in:.4f},{aupr_out:.4f},{fpr:.4f},{auroc_sc:.4f},{aupr_in_sc:.4f},{aupr_out_sc:.4f},{fpr_sc:.4f}".format(save_str=save_str,
                                                            auroc=m_dict_calibrated['AUROC'],
                                                            aupr_in=m_dict_calibrated['AUPR-IN'],
                                                            aupr_out=m_dict_calibrated['AUPR-OUT'],
                                                            fpr=m_dict_calibrated['FPR95TPR'],
                                                            auroc_sc=m_dict_unqs_scaled['AUROC'],
                                                            aupr_in_sc=m_dict_unqs_scaled['AUPR-IN'],
                                                            aupr_out_sc=m_dict_unqs_scaled['AUPR-OUT'],
                                                            fpr_sc=m_dict_unqs_scaled['FPR95TPR'])
        print("=> OOD Detection: ",save_str)
        with open(OOD_PATH,'a') as f:
            f.write("{}\n".format(save_str))
    print("************************************")
    print()
