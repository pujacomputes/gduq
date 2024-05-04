import pdb
import torch
import tqdm
from tap import Tap
from typing import List, Union
from typing import Optional, Sequence
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from models.encoders import GINEncoder, GINEncoderLayerwise, vGINEncoder, vGINEncoderLayerwise 
from GOOD.utils.train import nan2zero_get_mask


if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

def count_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])

class EvalArgs(Tap):
    ckpt_path: str = None
    ckpt_list: List[str] = None  # DEns 
    gde_ckpt_list: List[str] = None  # GDE 
    dropout_layers: List[int] = None  # GDE 
    uq_name = 'vanilla'
    save_results: int = 1

class dUQArgs(Tap):
    anchor_type: str = None
    gduq_type: str = "input"
    num_anchors: int = 10  # duq
    ckpt_path: str = None
    ckpt_list: List[str] = None
    gamma: float = 1.0
    layerwise_duq: int = 1
    save_results: int = 1
    uq_name = 'duq'
    save_path: str = "./ckpts/"


def test(model,loader,acc_metric,anchors=None,n_anchors=None):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad(): 
        for batch in tqdm.tqdm(loader,disable=True):
            if anchors is not None and n_anchors > 1:
                """
                By passing return_std, the model will return the calibrated logits! 
                """
                out,_ = model(batch.to(DEVICE),
                    anchors=anchors, 
                    n_anchors=n_anchors,
                    return_std=True)
            elif anchors is not None and n_anchors == 1 :
                """
                By passing return_std, the model will return the calibrated logits! 
                """
                out = model(batch.to(DEVICE),
                    anchors=anchors, 
                    n_anchors=n_anchors,
                    return_std=False)
            else:    
                out = model(batch.to(DEVICE))
            preds.append(out.to('cpu'))
            targets.append(batch.y.to('cpu'))

    preds = torch.cat(preds,dim=0) 
    targets = torch.cat(targets,dim=0)
    acc = acc_metric(preds,targets)
    return acc

def test_node(model,loader,acc_metric,config, anchors=None,split='val'):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad(): 
        for batch in tqdm.tqdm(loader,disable=True):
            batch=batch.to(DEVICE)
            mask, labels= nan2zero_get_mask(batch, split, config)
            if anchors is not None and config.uq.num_anchors > 1:
                """
                By passing return_std, the model will return the calibrated logits! 
                """
                out,_ = model(batch.to(DEVICE),
                    anchors=anchors, 
                    n_anchors=config.uq.num_anchors,
                    return_std=True)
            elif anchors is not None and config.uq.num_anchors == 1 :
                """
                By passing return_std, the model will return the calibrated logits! 
                """
                out = model(batch.to(DEVICE),
                    anchors=anchors, 
                    n_anchors=config.uq.num_anchors,
                    return_std=False) 
            else:
                out = model(batch)
            out = out[mask]
            preds.append(out.to('cpu'))
            targets.append(labels[mask].to('cpu'))
    preds = torch.cat(preds,dim=0)
    preds =  preds.argmax(dim=1)
    targets = torch.cat(targets,dim=0)
    acc = acc_metric(preds,targets.reshape(-1))
    return acc

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
            preds.append(out[labeled])
            targets.append(batch.y[labeled])

    preds = torch.cat(preds,dim=0).to('cpu') 
    targets = torch.cat(targets,dim=0).to('cpu')
    if anchors is not None and n_anchors > 1:
        unqs = torch.cat(unqs,dim=0).to('cpu')
        return preds, targets,unqs
    return preds,targets


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
        if uq_name == 'vanilla' or uq_name == 'temp':
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

    acc, cal_err_l1, cal_err_l2, cal_err_max = torch.Tensor([-1]), torch.Tensor([-1]), torch.Tensor([-1]), torch.Tensor([-1])
    try:
        cal_err_l1 = cal_metric_l1(preds,targets)
    except: 
        pdb.set_trace() 
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

def get_results_summary_node(model,loader,cal_metric_l1,cal_metric_l2, cal_metric_max, acc_metric,n_anchors=10, anchors=None, verbose=False,split='id_val'):

    """
    Get Calibration Error.
    """
    model.eval() 
    preds_list,targets_list,unqs_list = [],[],[]
    unqs=None
    for data in tqdm.tqdm(loader,disable=True):
        data = data.to(DEVICE)
        with torch.no_grad():
            if anchors is not None and n_anchors > 1:
                preds, unqs = model(data,anchors=anchors,n_anchors=n_anchors,return_std=True)
            elif anchors is not None and n_anchors == 1:
                preds = model(data,anchors=anchors,n_anchors=n_anchors,return_std=True)
                unqs = None
            else:
                preds  = model(data)
        preds_list.append(preds[data['{}_mask'.format(split)]].to('cpu')) 
        targets_list.append(data.y[data['{}_mask'.format(split)]].to('cpu'))
        # unqs_list.append(u[data['{}_mask'.format(split)]].mean(1).to('cpu'))
    
    preds= torch.cat(preds_list,dim=0)
    preds = torch.nn.functional.softmax(preds,dim=1) #this is done to rescale
    
    targets= torch.cat(targets_list,dim=0)
    # unqs= torch.cat(unqs_list,dim=0).mean() 

    acc, cal_err_l1, cal_err_l2, cal_err_max = torch.Tensor([-1]), torch.Tensor([-1]), torch.Tensor([-1]), torch.Tensor([-1])
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
    if verbose:
        print(stat_dict)
    return stat_dict