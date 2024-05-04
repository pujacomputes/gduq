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

from utils import (
    get_results_summary,
    DEVICE,
    EvalArgs,
    get_logits_labels,
    compute_temp,
    get_logits_labels_dens,
    count_parameters
)
from models.gduq_models import GraphANT,GraphANTHiddenReps, GraphANTLayerwiseGIN, GraphANTLayerwisevGIN 
from models.base_models import baseModel, baseModelvGIN, baseModelGINLayerwise, baseModelvGINLayerwise
from models.encoders import GINEncoder, GINEncoderLayerwise, vGINEncoder, vGINEncoderLayerwise 


from pytorch_ood.detector import MaxSoftmax
from pytorch_ood.utils import OODMetrics, ToUnknown
from torch_geometric.loader import DataLoader


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


def perform_voting_one_vs_one(model_list, loader, base_model_idx=0):
    """
    Get predictions.
    We treat one model as the pretrained model.
    All other models are considered the "check" members of the ensemble.
    We will compute the GDE score with each check member separately.
    We then average the disagreement scores to compute a threshold.
    e.g., what is the expected GDE score over the ensemble.

    base_model_idx tells us which model will be held out of the set of check models
    """
    with torch.no_grad():
        predictions = [[] for _ in model_list]
        total_correct = 0
        total_samples = 0
        target_list = []
        for x, y in loader:
            x = x.cuda()
            y = y.cuda()
            for enum, m in enumerate(model_list):
                pred = m(x).argmax(1)
                predictions[enum].append(pred)
                if enum == base_model_idx:
                    total_correct += pred.eq(y.data).sum().item()
                    total_samples += len(y)
                    target_list.append(y)
        for enum, p in enumerate(predictions):
            predictions[enum] = torch.cat(p)
    target_list = torch.cat(target_list).cpu()
    acc = total_correct / total_samples
    """
    Compute disagreement with all the other models
    """
    base_preds = predictions[base_model_idx]

    gde_disagreements = []
    for enum in range(len(model_list)):
        if enum != base_model_idx:
            check_preds = predictions[enum]
            gde_disagreements.append(base_preds.eq(check_preds))
    gde_disagreements = (
        torch.stack(gde_disagreements).float().cpu()
    )  # (num models-1) x n {0,1} binary array of GDE predictions
    """
    We will average the round-robin disagreement 
    scores to get a better score for each sample.
    """
    return gde_disagreements, predictions[base_model_idx].cpu(), target_list, acc


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
        acc_metric = BinaryAccuracy(multidim_average='global')
        gengap_acc_metric = BinaryAccuracy(
            multidim_average='global'
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
    print("=> SAVE NAME: ",save_name)
    print("=> MODEL PARAMS: ", count_parameters(model))

    if config.eval.uq_name.lower() == "temp":
        t = compute_temp(model, loader=loader["id_val"], temp=1, use_binary=use_binary)
        print("=> Temperature: ", np.round(t.cpu(), 4))
    else:
        t = torch.Tensor([1])
        print("=> Default Temperature: ", np.round(t, 4))

    LOG_PATH = "/usr/workspace/trivedi1/Fall2022/iclr22-graphduq-experiments/extended-gduq-eval/baseline_cal_logs.csv"
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

        else:
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

    """
    Generalization Gap
    Predict ID and OOD test and accuracy
    """

    # if args.predictor.lower() == "doc":
    GENGAP_DOC_PATH = "/usr/workspace/trivedi1/Fall2022/iclr22-graphduq-experiments/extended-gduq-eval/baseline_gengap_doc_logs.csv"
    if "doc" == "doc":
        if config.eval.uq_name.lower() == "dens":
            id_probs, id_labels = get_logits_labels_dens(
                model=model, loader=loader["id_val"], ckpt_list=config.eval.ckpt_list
            )
            target_probs, target_labels = get_logits_labels_dens(
                model=model, loader=loader["test"], ckpt_list=config.eval.ckpt_list
            )

            ood_val_probs, ood_val_labels = get_logits_labels_dens(
                model=model, loader=loader["val"], ckpt_list=config.eval.ckpt_list
            )

            id_test_probs, id_test_labels = get_logits_labels_dens(
                model=model, loader=loader["id_test"], ckpt_list=config.eval.ckpt_list
            )

        else:
            id_probs, id_labels = get_logits_labels(
                model=model, loader=loader["id_val"]
            )
            target_probs, target_labels = get_logits_labels(
                model=model,
                loader=loader["test"],
            )

            ood_val_probs, ood_val_labels = get_logits_labels(
                model=model, loader=loader["val"]
            )

            id_test_probs, id_test_labels = get_logits_labels(
                model=model,
                loader=loader["id_test"],
            )

        if config.dataset.num_classes > 2:
            id_probs = torch.nn.functional.softmax(id_probs, dim=1)
            id_confs = id_probs.max(dim=1)[0]
            id_preds = id_probs.argmax(dim=1)

            target_probs = torch.nn.functional.softmax(target_probs, dim=1)
            target_confs = target_probs.max(dim=1)[0]
            target_preds = target_probs.argmax(dim=1)

            ood_val_probs = torch.nn.functional.softmax(ood_val_probs, dim=1)
            ood_val_confs = ood_val_probs.max(dim=1)[0]
            ood_val_preds = ood_val_probs.argmax(dim=1)

            id_test_probs = torch.nn.functional.softmax(id_test_probs, dim=1)
            id_test_confs = id_test_probs.max(dim=1)[0]
            id_test_preds = id_test_probs.argmax(dim=1)
        else:
            id_probs = torch.sigmoid(id_probs)
            id_confs = torch.maximum(id_probs, 1 - id_probs)
            id_preds = (id_probs > 0.5).float()

            target_probs = torch.nn.functional.sigmoid(target_probs)
            target_confs = torch.maximum(target_probs, 1 - target_probs)
            target_preds = (target_probs > 0.5).float()

            ood_val_probs = torch.nn.functional.sigmoid(ood_val_probs)
            ood_val_confs = torch.maximum(ood_val_probs, 1 - ood_val_probs)
            ood_val_preds = (ood_val_probs > 0.5).float()

            id_test_probs = torch.nn.functional.sigmoid(id_test_probs)
            id_test_preds = (id_test_probs > 0.5).float()
            id_test_confs = torch.maximum(id_test_probs, 1 - id_test_probs)

        id_val_acc = gengap_acc_metric(id_preds, id_labels).item()
        ood_test_acc = gengap_acc_metric(target_preds, target_labels).item()
        ood_val_acc = gengap_acc_metric(ood_val_preds, ood_val_labels).item()
        id_test_acc = gengap_acc_metric(id_test_preds, id_test_labels).item()

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

        pred_target_acc_w_id_thres = (target_confs >= id_val_found_thres).sum() / len(
            target_confs
        )
        pred_target_acc_w_ood_thres = (target_confs >= ood_val_found_thres).sum() / len(
            target_confs
        )
        pred_id_test_acc_w_id_thres = (id_test_confs >= id_val_found_thres).sum() / len(
            id_test_confs
        )

        mae_diff_w_id_thres = np.abs(ood_test_acc - pred_target_acc_w_id_thres)
        mae_diff_id_test_w_id_thres = np.abs(id_test_acc - pred_id_test_acc_w_id_thres)
        mae_diff_w_ood_thres = np.abs(ood_test_acc - pred_target_acc_w_ood_thres)

        save_str_id = "{id_thres:.4f},{id_pred_acc:.4f},{id_true_acc:.4f},{id_mae:.4f},{ood_pred:.4f}, {ood_true_acc:.4f}, {ood_mae:.4f},{ood_thres:.4f}, {ood_pred_ood_acc:.3f}, {ood_ood_true_acc:.3f}, {ood_ood_mae:.3f} ".format(
            id_thres=id_val_found_thres,
            id_pred_acc=pred_id_test_acc_w_id_thres,
            id_true_acc=id_test_acc,
            id_mae=mae_diff_id_test_w_id_thres,
            ood_pred=pred_target_acc_w_id_thres,
            ood_true_acc=ood_test_acc,
            ood_mae=mae_diff_w_id_thres,
            mae=mae_diff_w_id_thres,
            ood_thres=ood_val_found_thres,
            ood_pred_ood_acc=pred_target_acc_w_ood_thres,
            ood_ood_true_acc=ood_test_acc,
            ood_ood_mae=mae_diff_w_ood_thres,
        )

        print("=> DOC: ", save_str_id)
        if config.eval.save_results > 0:
            with open(GENGAP_DOC_PATH, "a") as f:
                f.write("{},doc,{}\n".format(save_name, save_str_id))

    if config.eval.gde_ckpt_list is None:
        print("=> ONLY 1 MODEL PASSED. NOT SUFFICIENT for MDE. SKIPPING")
    else:
        """
        Get base model's prediction's
        Determine if number of disagreements.
        Then, return the percentage of disagreements.
        """
        pred_list = []
        target_list = []
        for ckpt_p in config.eval.gde_ckpt_list:
            ckpt = torch.load(ckpt_p)
            model.encoder.load_state_dict(ckpt["encoder"])
            model.classifier.load_state_dict(ckpt["classifer"])
            model.eval()
            loader_preds = []
            loader_targets = []
            for l_x in ["id_test", "test"]:
                p, t = get_logits_labels(model=model, loader=loader[l_x])
                if config.dataset.num_classes > 2:
                    probs = torch.nn.functional.softmax(p, dim=1)
                    preds = probs.argmax(dim=1)
                else:
                    probs = torch.nn.functional.sigmoid(p)
                    preds = (probs > 0.5).float()

                loader_preds.append(preds)
                loader_targets.append(t)
            pred_list.append(loader_preds)
            target_list.append(loader_targets)
        # NumEns, samples, num-classes

        ### Compute the disagreements
        id_pred_acc = pred_list[0][0].eq(pred_list[1][0]).sum() / len(pred_list[0][0])
        id_true_acc = pred_list[0][0].eq(target_list[0][0]).sum() / len(pred_list[0][0])
        id_mae = np.abs(id_true_acc.item() - id_pred_acc.item())

        ood_pred_acc = pred_list[0][1].eq(pred_list[1][1]).sum() / len(pred_list[0][1])
        ood_true_acc = pred_list[0][1].eq(target_list[0][1]).sum() / len(
            pred_list[0][1]
        )
        ood_mae = np.abs(ood_true_acc.item() - ood_pred_acc.item())

        save_str_id = "{id_pred_acc:.4f},{id_true_acc:.4f},{id_mae:.4f},{ood_pred:.4f}, {ood_true_acc:.4f}, {ood_mae:.4f}".format(
            id_pred_acc=id_pred_acc,
            id_true_acc=id_true_acc,
            id_mae=id_mae,
            ood_pred=ood_pred_acc,
            ood_true_acc=ood_true_acc,
            ood_mae=ood_mae,
        )

        GENGAP_GDE_PATH = "/usr/workspace/trivedi1/Fall2022/iclr22-graphduq-experiments/extended-gduq-eval/baseline_gengap_gde_logs.csv"
        print("=> GDE: ", save_str_id)
        if config.eval.save_results > 0:
            with open(GENGAP_GDE_PATH, "a") as f:
                f.write("{},gde,{}\n".format(save_name, save_str_id))

    """
    OOD Detection
    """
    print("************************************")
    print()
    OOD_LOG_PATH = "/usr/workspace/trivedi1/Fall2022/iclr22-graphduq-experiments/extended-gduq-eval/baseline_ood_logs.csv"
    id_val_loader = loader["id_val"]
    for key in sorted(["val", "test"]):
        tmp_loader = loader[key]
        to_u = ToUnknown()
        tmp_dataset = [to_u(d) for d in tmp_loader.dataset]
        tmp_loader = DataLoader(tmp_dataset)
        """
        DENs
        """
        if config.eval.uq_name.lower() == "dens":
            pred_list = []
            targets = []
            for ckpt_p in config.eval.ckpt_list:
                ckpt = torch.load(ckpt_p)
                model.encoder.load_state_dict(ckpt["encoder"])
                model.classifier.load_state_dict(ckpt["classifer"])
                model.eval()
                ps = []
                ts = []
                for l_x in [id_val_loader, tmp_loader]:
                    p, t = get_logits_labels(model=model, loader=l_x)
                    ps.append(p)
                    ts.append(t)
                if config.dataset.num_classes > 2:
                    preds = torch.vstack(ps)
                else:
                    preds = torch.cat(ps)

                targets = torch.cat(ts)

                pred_list.append(preds)
            # NumEns, samples, num-classes

            preds = torch.stack(pred_list, dim=0).mean(dim=0)
            # targets = t
            # Take the Mean.
            #
        else:
            id_p, id_t = get_logits_labels(model=model, loader=id_val_loader)
            ood_p, ood_t = get_logits_labels(model=model, loader=tmp_loader)

            if config.dataset.num_classes > 2:
                preds = torch.vstack([id_p, ood_p])
            else:
                preds = torch.cat([id_p, ood_p])
            targets = torch.cat([id_t, ood_t])

        if config.dataset.num_classes > 2:
            detector = MaxSoftmax(model)
        else:
            detector = MaxSigmoid(model)

        metrics = OODMetrics()
        metrics.reset()

        preds = detector.score(preds)
        metrics.update(preds, targets)
        m_dict = metrics.compute()

        save_str = ",".join([save_name, key])
        save_str = (
            "{save_str},{auroc:.4f},{aupr_in:.4f},{aupr_out:.4f},{fpr:.4f}".format(
                save_str=save_str,
                auroc=m_dict["AUROC"],
                aupr_in=m_dict["AUPR-IN"],
                aupr_out=m_dict["AUPR-OUT"],
                fpr=m_dict["FPR95TPR"],
            )
        )
        print("=> OOD: ", save_str)
        if config.eval.save_results > 0:
            with open(OOD_LOG_PATH, "a") as f:
                f.write("{}\n".format(save_str))
    print("************************************")
    print()
