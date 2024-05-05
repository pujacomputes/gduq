# [Accurate and Scalable Estimation of Epistemic Uncertainty for Graph Neural Networks](https://arxiv.org/abs/2401.03350v1#S8)

![Overview of GDUQ](assets/Overview-of-GDUQ.png)

## Abstract
While graph neural networks (GNNs) are widely used for node and graph representation learning tasks, the reliability of GNN uncertainty estimates under distribution shifts remains relatively under-explored. Indeed, while \textit{post-hoc} calibration strategies can be used to improve in-distribution calibration, they need not also improve calibration under distribution shift. However, techniques which produce GNNs with better \textit{intrinsic} uncertainty estimates are particularly valuable, as they can always be combined with post-hoc strategies later. Therefore, in this work, we propose G-$\Delta$UQ, a novel training framework designed to improve intrinsic GNN uncertainty estimates. Our framework adapts the principle of stochastic data centering to graph data through novel graph anchoring strategies, and is able to support partially stochastic GNNs. While, the prevalent wisdom is that fully stochastic networks are necessary to obtain reliable estimates, we find that the functional diversity induced by our anchoring strategies when sampling hypotheses renders this unnecessary and allows us to support G-$\Delta$UQ on pretrained models. Indeed, through extensive evaluation under covariate, concept and graph size shifts, we show that G-$\Delta$UQ leads to better calibrated GNNs for node and graph classification. Further, it also improves performance on the uncertainty-based tasks of out-of-distribution detection and generalization gap estimation. Overall, our work provides insights into uncertainty estimation for GNNs, and demonstrates the utility of G-$\Delta$UQ in obtaining reliable estimates.


## Installation/Requirements

We use cuda 11.8.0. 
```
(cuda 11.8.0)
GOOD (https://github.com/divelab/GOOD?tab=readme-ov-file#installation)
torchmetrics
torch
tqdm
prettyprint
torch_geometric
munch
typed-argument-parser
pytorch_ood
```

If you have GLIBC error when installing torch-sparse or pyg-lib, try installing from source. 
## Source Code

Please see `scripts/` for example scripts on how to run. ⚠️ This repo is currently being developed, so expect some more scripts and details.

## Citation

```
@inproceedings{
trivedi2024accurate,
title={Accurate and Scalable Estimation of Epistemic Uncertainty for Graph Neural Networks},
author={Puja Trivedi and Mark Heimann and Rushil Anirudh and Danai Koutra and Jayaraman J. Thiagarajan},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=ZL6yd6N1S2}
}
```

## Acknowledgements 
Thanks to [CaGCN](https://proceedings.neurips.cc/paper/2021/hash/c7a9f13a6c0940277d46706c7ca32601-Abstract.html) and the [GOOD benchmark](https://github.com/divelab/GOOD/) for providing post hoc calibration techniques and graph OOD benchmark datasets, respectively! 

