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
from GOOD.utils.args import args_parser
from GOOD.utils.config_reader import load_config, args2config, process_configs

from torchmetrics.classification import (
    BinaryCalibrationError,
    MulticlassCalibrationError,
    BinaryAccuracy,
    MulticlassAccuracy,
    BinaryAUROC,
)
import sys
sys.path.append("/p/lustre3/trivedi1/GDUQ/src")

from calibration.calib_utils import (
    get_results_summary,
    DEVICE,
    EvalArgs,
    get_logits_labels,
    compute_temp,
    get_logits_labels_dens,
)

from pytorch_ood.detector import MaxSoftmax
from pytorch_ood.utils import OODMetrics, ToUnknown
from torch_geometric.loader import DataLoader
from calibration.graph_calibrator import ETS,TS,VS,IRM, Dirichlet, OrderInvariantCalib, SplineCalib


from models.base_models import baseModel
from models.encoders import GINEncoder, GINEncoderLayerwise, vGINEncoder, vGINEncoderLayerwise 

class MaxSigmoid:
    def __init__(self, net) -> None:
        self.net = net

    def score(self, raw_logits):
        sig_logits = torch.nn.functional.sigmoid(raw_logits)
        sig_conf = torch.maximum(sig_logits, 1 - sig_logits)
        return sig_conf


class ToUnknown(object):
    def __call__(self, data):
        # this is done for the OOD dataset.
        desired_shape = data.y.shape
        data.y = torch.LongTensor([-1]).reshape(desired_shape)
        return data


if __name__ == "__main__":
    """
    Setup Model
    """

    args = args_parser()
    config, duplicate_warnings, duplicate_errors = load_config(args.config_path)
    print("=> Duplicate Warnings: ", duplicate_warnings)
    print("=> Duplicate Errors: ", duplicate_errors)

    args2config(config, args)
    config = munchify(config)
    process_configs(config)

    eval_args = EvalArgs(explicit_bool=True).parse_args(known_only=True)
    config.eval = munchify(eval_args)

    """
    Load Data 
    """
    dataset = load_dataset(config.dataset.dataset_name, config)
    loader = create_dataloader(dataset, config)  # TURN OFF SHUFFLING!
    _ = loader.pop("train", None)
    id_val = loader["id_val"]
    num_feats = dataset["train"][0].x.shape[1]
    print("=> Num Feats: ", num_feats)
    inputs = next(iter(id_val))
    inputs = inputs.to(DEVICE)
    print("=> Input Shape: ", inputs.x.shape)

    """
    Create Anchor Distribution
    """
    config.dataset.dim_node = dataset["train"][0].x.shape[1]  # intentional [x-c,c]
    config.model.model_level = "graph"

    """
    Create Model.
    """
    if config.model.model_name == "vGIN":
        print("=> Using vGIN")
        gin_enc = vGINEncoder(config)
    else:
        gin_enc = GINEncoder(config)
        print("=> Using GIN")

    model = baseModel(encoder=gin_enc, num_classes=config.dataset.num_classes)
    model.to(DEVICE)
    model.eval()

    if config.eval.uq_name.lower() == "dens":
        config.eval.ckpt_path = config.eval.ckpt_list[0]
    if config.eval.ckpt_path is None:
        config.eval.ckpt_path = config.eval.ckpt_list[0]
    ckpt = torch.load(config.eval.ckpt_path)
    model.encoder.load_state_dict(ckpt["encoder"])
    model.classifier.load_state_dict(ckpt["classifer"])
    print("=> Model Loaded! ")
    """
    Load Data 
    """
    sample = next(iter(id_val))
    sample = sample.to(DEVICE)
    print("=> Input Shape: ", inputs.x.shape)

    """
    Set-up Metrics
    """
    if config.dataset.num_classes <= 2:
        cal_metric_l1 = BinaryCalibrationError(n_bins=100, norm="l1")
        cal_metric_l2 = BinaryCalibrationError(n_bins=100, norm="l2")  # .to(DEVICE)
        cal_metric_max = BinaryCalibrationError(n_bins=100, norm="max")  # .to(DEVICE)
        acc_metric = BinaryAccuracy(
           multidim_average="global"
        )
        gengap_acc_metric = BinaryAccuracy(
           multidim_average="global"
        )
        use_binary = True
    else:
        cal_metric_l1 = MulticlassCalibrationError(
            num_classes=config.dataset.num_classes, n_bins=100, norm="l1"
        )  # .to(DEVICE)
        cal_metric_l2 = MulticlassCalibrationError(
            num_classes=config.dataset.num_classes, n_bins=100, norm="l2"
        )  # .to(DEVICE)
        cal_metric_max = MulticlassCalibrationError(
            num_classes=config.dataset.num_classes, n_bins=100, norm="max"
        )  # .to(DEVICE)
        acc_metric = MulticlassAccuracy(
            num_classes=config.dataset.num_classes, average="micro"
        )
        gengap_acc_metric = MulticlassAccuracy(
            num_classes=config.dataset.num_classes, average="micro"
        )
        use_binary = False

    save_name = "_".join(
        [
            config.eval.uq_name,
            config["dataset"]["dataset_name"],
            config["dataset"]["domain"],
            config["dataset"]["shift_type"],
            config["model"]["model_name"],
            str(config["random_seed"]),
        ]
    )

    t = torch.Tensor([1])
    if config.eval.uq_name.upper() in ['TS', 'VS', 'ETS']:
        cal_wdecay = 0
    elif config.eval.uq_name.upper() == 'CaGCN':
        if config.dataset.dataset_name == "CoraFull":
            cal_wdecay = 0.03
        else:
            cal_wdecay = 5e-3
    else:
        cal_wdecay = 5e-4
    if config.eval.uq_name.lower() == "temp":
        t = compute_temp(model, loader=loader["id_val"], temp=1, use_binary=use_binary)
        print("=> Temperature: ", np.round(t.cpu(), 4))
    
    elif config.eval.uq_name.lower() == "ets":
        temp_model = ETS(model, config.dataset.num_classes)
        temp_model.fit(loader['id_val'], loader['eval_train'], cal_wdecay)
    elif config.eval.uq_name.lower() == "ts":
        temp_model = TS(model)
        temp_model.fit(loader['id_val'], loader['eval_train'], cal_wdecay)
    elif config.eval.uq_name.lower() == "vs":
        temp_model = VS(model,config.dataset.num_classes)
        temp_model.fit(loader['id_val'], loader['eval_train'], cal_wdecay)
    elif config.eval.uq_name.lower() == "irm":
        temp_model = IRM(model)
        temp_model.fit(loader['id_val'], loader['eval_train'], cal_wdecay)
    elif config.eval.uq_name.lower() == "dirichlet":
        temp_model = Dirichlet(model,config.dataset.num_classes)
        temp_model.fit(loader['id_val'], loader['eval_train'], cal_wdecay)
    elif config.eval.uq_name.lower() == "spline":
        temp_model = SplineCalib(model,7)
        temp_model.fit(loader['id_val'], loader['eval_train'], cal_wdecay)
    elif config.eval.uq_name.lower() == "orderinvariant":
        temp_model = OrderInvariantCalib(model,config.dataset.num_classes)
        temp_model.fit(loader['id_val'], loader['eval_train'], cal_wdecay)
    else:
        t = torch.Tensor([1])
        print("=> Default Temperature: ", np.round(t, 4))

    LOG_PATH = "logs.csv"
    loader.pop("eval_train")
    print("====================================")
    for key in sorted(loader.keys()):
        tmp_loader = loader[key]

        """
        DENs
        """
        if config.eval.uq_name.lower() == "dens":
            preds, targets = get_logits_labels_dens(model=model,
                                                    loader=tmp_loader,
                                                    ckpt_list =config.eval.ckpt_list)
            cal_err_l1 = cal_metric_l1(preds, targets)
            cal_err_l2 = cal_metric_l2(preds, targets)
            cal_err_max = cal_metric_max(preds, targets)
            acc = acc_metric(preds, targets)
            stat_dict = {
                "acc": acc,
                "cal_err_l1": cal_err_l1,
                "cal_err_l2": cal_err_l2,
                "cal_err_max": cal_err_max,
            }
        elif config.eval.uq_name.lower() in ["ets","ts","vs","irm","dirichlet","spline","orderinvariant"]:
            stat_dict = get_results_summary(
                            temp_model,
                            loader=tmp_loader,
                            cal_metric_l1=cal_metric_l1,
                            cal_metric_l2=cal_metric_l2,
                            cal_metric_max=cal_metric_max,
                            acc_metric=acc_metric,
                            uq_name=config.eval.uq_name,
                            t=t,
                        )
        else:
            print("YOU MADE A MISTAKE.")
            exit()
            stat_dict = get_results_summary(
                model,
                loader=tmp_loader,
                cal_metric_l1=cal_metric_l1,
                cal_metric_l2=cal_metric_l2,
                cal_metric_max=cal_metric_max,
                acc_metric=acc_metric,
                uq_name=config.eval.uq_name,
                t=t,
            )

        acc = str(np.round(stat_dict["acc"].item(), 4))
        cal_err_l1 = str(np.round(stat_dict["cal_err_l1"].item(), 4))
        cal_err_l2 = str(np.round(stat_dict["cal_err_l2"].item(), 4))
        cal_err_max = str(np.round(stat_dict["cal_err_max"].item(), 4))

        save_str = ",".join([save_name, key, acc, cal_err_l1, cal_err_l2, cal_err_max])
        print("=> {}".format(save_str))

        if config.eval.save_results > 0:
            with open(LOG_PATH, "a") as f:
                f.write("{}\n".format(save_str))
    print("************************************")
    print()

   
