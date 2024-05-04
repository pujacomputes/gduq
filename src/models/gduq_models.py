import torch
import numpy as np
import pdb

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


class GraphANT(torch.nn.Module):
    def __init__(self, base_network, mean, std, anchor_type="graph", num_classes=10):
        super(GraphANT, self).__init__()
        """
            base_network (default: None):
                network used to perform anchor training.
                first GNN layer should be modified to accept 2x the node features.     
            """
        if base_network is not None:
            self.net = base_network
        else:
            raise Exception("base network needs to be defined")

        self.mean = mean
        self.std = std
        """
            we will sample anchors from this distribution. Anchors can be: 
            -- NODE-specific (different anchor per node)
            -- GRAPH-specific (different anchor per graph) 
            """
        assert anchor_type.lower() in [
            "node",
            "graph",
            "batch",
            "graph2graph",
            "shuffle",
            "debug",
            "random"
        ]
        self.anchor_type = anchor_type.lower()

        if self.mean is not None and self.std is not None:
            self.anchor_dist = torch.distributions.normal.Normal(
                loc=self.mean, scale=self.std
            )
        self.num_classes = num_classes

    """
    This function draws a fixed set of base anchors 
    for inference. At inference time, we must use "node" anchoring.
    Therefore, we return [num_anchors,node_features].
    Process batch will have to subtract this value from each node accordingly.
    """

    def get_anchors(self, batch, num_anchors=1):
        anchor_list = []
        if self.anchor_type == "graph2graph":
            batch = batch.to_data_list()
        for i in range(num_anchors):
            if self.anchor_type == "graph2graph":
                anchors = batch[i].x.unsqueeze(
                    0
                )  # select anchors from batch, assumes the same number of nodes!!
            elif self.anchor_type == "shuffle":
                # shuffle the copied batch along the nodes
                copied_batch = batch.x.clone().detach()
                rand_idx = torch.randperm(batch.x.shape[0])
                anchors = copied_batch[rand_idx]
            elif self.anchor_type == "debug":
                anchors = torch.zeros_like(batch.x)
            else:
                anchors = self.anchor_dist.sample([batch.x.shape[0]])
            anchor_list.append(anchors)
        anchors = torch.cat(anchor_list, dim=0)
        return anchors.to(DEVICE)

    def process_batch(self, batch, anchors):
        """
        anchors (default=None):
            if passed, will use the same set of anchors for all batches during training.
            if  None, we will randomly draw samples from the anchor dist.
        """
        if anchors == None:
            if self.anchor_type == "node":  # anchors: [num_nodes, num_feats]
                anchors = self.anchor_dist.sample([batch.x.shape[0]]).to(DEVICE)
                new_feats = torch.cat((batch.x - anchors, anchors), dim=1)
            elif self.anchor_type == "batch":
                anchors = self.anchor_dist.sample([1]).to(DEVICE)
                new_feats = torch.cat(
                    (batch.x - anchors, anchors.repeat(batch.x.shape[0], 1)), dim=1
                )
            elif self.anchor_type == "graph":
                # get nodes in each graph
                counts = batch.batch.unique(return_counts=True)[1]
                anchors = self.anchor_dist.sample([batch.num_graphs]).to(DEVICE)
                anchors = torch.repeat_interleave(anchors, counts, dim=0).to(
                    DEVICE
                )  # repeat anchor to match number of nodes in each graph
                new_feats = torch.cat((batch.x - anchors, anchors), dim=1)
            elif self.anchor_type == "graph2graph":
                # assumes the same number of nodes
                # this means we can stack the anchor to match, instead of first converting to a datalist
                batch_list = batch.to_data_list()
                batch_order = np.arange(batch.num_graphs)
                np.random.shuffle(batch_order)
                anchors = torch.vstack([batch_list[i].x for i in batch_order])
                new_feats = torch.cat((batch.x - anchors, anchors), dim=1)
            elif self.anchor_type == "shuffle":
                copied_batch = batch.x.clone().detach()
                rand_idx = torch.randperm(batch.x.shape[0])
                anchors = copied_batch[rand_idx]
                new_feats = torch.cat((batch.x - anchors, anchors), dim=1)
            elif self.anchor_type == "debug":
                anchors = torch.zeros_like(batch.x).to(DEVICE)
                new_feats = torch.cat((batch.x - anchors, anchors), dim=1)
        else:
            """
            This is used for inference time.
            And is "node" based anchoring b/c
            we need the same anchor across different sized graphs.
            """
            if self.anchor_type == "graph2graph":
                # assumes same number of nodes per graph.
                anchors = torch.repeat_interleave(
                    anchors.squeeze(0), batch.num_graphs, dim=0
                ).to(DEVICE)
                new_feats = torch.cat((batch.x - anchors, anchors), dim=1)
            elif self.anchor_type == "debug":
                anchors = torch.zeros_like(batch.x).to(DEVICE)
                new_feats = torch.cat((batch.x - anchors, anchors), dim=1)
            elif self.anchor_type == "shuffle":
                copied_batch = batch.x.clone().detach()
                rand_idx = torch.randperm(batch.x.shape[0])
                anchors = copied_batch[rand_idx]
                new_feats = torch.cat((batch.x - anchors, anchors), dim=1)
            else:
                anchors = torch.repeat_interleave(anchors, batch.x.shape[0], dim=0).to(
                    DEVICE
                )
                new_feats = torch.cat((batch.x - anchors, anchors), dim=1)

        return new_feats.to(DEVICE)

    def calibrate(self, mu, sig):
        """
        For ImageNet we use mu_hat = mu/c
        For CIFAR10/100 we use mu_hat = mu/(1+exp(c))
        """
        c = torch.mean(
            sig, dim=1, keepdim=True
        )  # batch-size,num-classes => batch-size,1
        c = c.expand(mu.shape)  # batch-size,1 =>
        return torch.div(mu, 1 + torch.exp(c))
        # return torch.div(mu,c)

    def forward(self, x, anchors=None, n_anchors=1, return_std=False,return_unq=False):
        if n_anchors == 1 and return_std:
            raise Warning("Use n_anchor>1, std. dev cannot be computed!")

        preds = torch.zeros(n_anchors, x.num_graphs, self.num_classes).to(DEVICE)
        for n in range(n_anchors):
            with torch.no_grad():
                if anchors == None:
                    new_feats = self.process_batch(x, anchors=None)
                else:
                    new_feats = self.process_batch(x, anchors=anchors[n].unsqueeze(0))
            preds[n, :, :] = self.net.forward_graph(new_feats, x.edge_index, x.batch)
        mu = preds.mean(dim=0)
        if return_std:
            std = preds.sigmoid().std(dim=0)
            # std = preds.std(dim=0)
            return self.calibrate(mu, std), std
        else:
            return mu


class GraphANTHiddenReps(torch.nn.Module):
    def __init__(self, base_network, mean, std, anchor_type="graph", num_classes=10):
        super(GraphANTHiddenReps, self).__init__()
        """
            base_network (default: None):
                network used to perform anchor training.
                first GNN layer should be modified to accept 2x the node features.     
            """
        if base_network is not None:
            self.net = base_network
        else:
            raise Exception("base network needs to be defined")

        self.mean = mean
        self.std = std
        """
            We will create hidden-representation space graph anchors.
            They can randomly be defined or defined through other samples 
            in the batch (default behavior). 
            """
        assert anchor_type.lower() in ["random", "batch"]
        self.anchor_type = anchor_type.lower()
        # self.anchor_dist = torch.distributions.normal.Normal(
        #     loc=self.mean, scale=self.std
        # )
        self.num_classes = num_classes
        self.hidden_rep_dim = 300

    """
    This function draws a fixed set of base anchors 
    for inference. Since we are performing anchoring 
    at the hidden representation space, we will return 
    [num_anchors,hidden-rep-dim].
    Process batch will have to subtract this value from each hidden rep accordingly.
    """

    def get_anchors(self, batch, num_anchors=1):
        with torch.no_grad():
            if self.anchor_type == "batch":
                anchors = self.net.forward_features(batch)[0:num_anchors, :]
            elif self.anchor_type == "random":
                anchors = self.anchor_dist.sample([num_anchors])
            else:
                print("INVALID ANCHOR TYPE")
                exit()
        return anchors.to(DEVICE)

    def process_batch(self, hidden_rep, anchors):
        """
        hidden_rep [num_graphs, hidden_rep_dim]: anchoring occurs before the classifier,
        so we pass the hidden rep here (not the original batch). [num_graphs, hidden_rep_dim]
        batch: we pass the original data batch too. This is needed so that we can pull out the anchors
        anchors (default=None):
            if an anchor (1) is passed, then samples in the batch will share this anchor. (inference default)
            if None, we will create a separate anchor per sample. (training default)
        """
        if anchors == None:
            if self.anchor_type == "random":
                # anchors: [num_graphs, hidden_rep_dim]
                anchors = self.anchor_dist.sample([hidden_rep.shape[0]]).to(DEVICE)
                new_hidden_rep = torch.cat((hidden_rep - anchors, anchors), dim=1)
            elif self.anchor_type == "batch":
                # returns a cloned/detached version of the anchors
                # that are shuffled.
                anchors = hidden_rep.clone().detach()[
                    torch.randperm(hidden_rep.shape[0])
                ]
                new_hidden_rep = torch.cat((hidden_rep - anchors, anchors), dim=1)
        else:
            anchors = torch.repeat_interleave(anchors, hidden_rep.shape[0], dim=0).to(
                DEVICE
            )
            new_hidden_rep = torch.cat((hidden_rep - anchors, anchors), dim=1)

        return new_hidden_rep.to(DEVICE)

    def calibrate(self, mu, sig):
        """
        For ImageNet we use mu_hat = mu/c
        For CIFAR10/100 we use mu_hat = mu/(1+exp(c))
        """
        c = torch.mean(
            sig, dim=1, keepdim=True
        )  # batch-size,num-classes => batch-size,1
        c = c.expand(mu.shape)  # batch-size,1 =>
        return torch.div(mu, 1 + torch.exp(c))
        # return torch.div(mu,c)

    """
    This computation graph is more 
    complicated than the original delta-UQ formulation.
    Gradients will be propagated through the residual operation.
    However, they should not be propagated through the hidden anchor representation
    (so we are not computing the gradient twice).
    """

    def forward(self, batch, anchors=None, n_anchors=1, return_std=False,return_unq=False):
        if n_anchors == 1 and return_std:
            raise Warning("Use n_anchor>1, std. dev cannot be computed!")

        preds = torch.zeros(n_anchors, batch.num_graphs, self.num_classes).to(DEVICE)
        clean_hidden_reps = self.net.forward_features(batch)
        for n in range(n_anchors):
            if anchors == None:
                new_feats = self.process_batch(clean_hidden_reps, anchors=None)
            else:
                new_feats = self.process_batch(
                    clean_hidden_reps, anchors=anchors[n].unsqueeze(0)
                )
            preds[n, :, :] = self.net.forward_classifier(new_feats)
        mu = preds.mean(dim=0)
        if return_std:
            std = preds.sigmoid().std(dim=0)
            # std = preds.std(dim=0)
            return self.calibrate(mu, std), std
        else:
            return mu

    def forward_preds(self, batch, anchors=None, n_anchors=1, return_std=False,return_unq=False):
        if n_anchors == 1 and return_std:
            raise Warning("Use n_anchor>1, std. dev cannot be computed!")

        preds = torch.zeros(n_anchors, batch.num_graphs, self.num_classes).to(DEVICE)
        clean_hidden_reps = self.net.forward_features(batch)
        for n in range(n_anchors):
            if anchors == None:
                new_feats = self.process_batch(clean_hidden_reps, anchors=None)
            else:
                new_feats = self.process_batch(
                    clean_hidden_reps, anchors=anchors[n].unsqueeze(0)
                )
            preds[n, :, :] = self.net.forward_classifier(new_feats)
        return preds

    def update_anchor_dist(self,dataloader,num_nodes,x_dim):

        x = torch.zeros(num_nodes, x_dim) 
        with torch.no_grad():
            start_idx = end_idx = 0
            for d in dataloader:
                d = d.to(DEVICE) 
                end_idx = start_idx + int(max(d.batch)+1)
                # end_idx = start_idx + d.x.shape[0]
                x[start_idx:end_idx,:] = self.net.forward_features(d).cpu()
                start_idx = end_idx
        mu = x.mean(dim=0)
        std = x.std(dim=0)
        std_mean = std.mean()
        std[std == 0] = std_mean
        
        self.mean = x.mean(dim=0)
        self.std = std
        print("Updated Mu: ", mu.shape)
        print("Updated Std: ", std.min(), std.max(), std_mean)
        assert std_mean > 0
        self.anchor_dist = torch.distributions.normal.Normal(
            loc=self.mean, scale=self.std
        )

    def forward_pretrain(self, batch, anchors=None, n_anchors=1, return_std=False,return_unq=False):
        if n_anchors == 1 and return_std:
            raise Warning("Use n_anchor>1, std. dev cannot be computed!")

        preds = torch.zeros(n_anchors, batch.num_graphs, self.num_classes).to(DEVICE)
        clean_hidden_reps = self.net.forward_features(batch)
        for n in range(n_anchors):
            with torch.no_grad():
                if anchors == None:
                    new_feats = self.process_batch(clean_hidden_reps, anchors=None)
                else:
                    new_feats = self.process_batch(
                        clean_hidden_reps, anchors=anchors[n].unsqueeze(0)
                    )
            preds[n, :, :] = self.net.forward_classifier(new_feats)
        mu = preds.mean(dim=0)
        if return_std:
            std = preds.sigmoid().std(dim=0)
            return self.calibrate(mu, std), std
        else:
            return mu

class GraphANTNode(torch.nn.Module):
    def __init__(self, base_network, mean, std, anchor_type="graph", num_classes=10):
        super(GraphANTNode, self).__init__()
        """
            base_network (default: None):
                network used to perform anchor training.
                first GNN layer should be modified to accept 2x the node features.     
            """
        if base_network is not None:
            self.net = base_network
        else:
            raise Exception("base network needs to be defined")

        self.mean = mean
        self.std = std
        """
            we will sample anchors from this distribution. Anchors can be: 
            -- NODE-specific (different anchor per node)
            -- GRAPH-specific (different anchor per graph) 
            """
        assert anchor_type.lower() in [
            "node",
            "graph",
            "batch",
            "graph2graph",
            "shuffle",
            "debug",
        ]
        self.anchor_type = anchor_type.lower()
        self.anchor_dist = torch.distributions.normal.Normal(
            loc=self.mean, scale=self.std
        )
        self.num_classes = num_classes

    """
    This function draws a fixed set of base anchors 
    for inference. At inference time, we must use "node" anchoring.
    Therefore, we return [num_anchors,node_features].
    Process batch will have to subtract this value from each node accordingly.
    """

    def get_anchors(self, batch, num_anchors=1):
        anchor_list = []
        if self.anchor_type == "graph2graph":
            batch = batch.to_data_list()
        for i in range(num_anchors):
            if self.anchor_type == "graph2graph":
                anchors = batch[i].x.unsqueeze(
                    0
                )  # select anchors from batch, assumes the same number of nodes!!
            elif self.anchor_type == "shuffle":
                copied_batch = batch.x.clone().detach()
                # shuffle the copied  batch along x
                rand_idx = torch.randperm(batch.x.shape[0])
                anchors = copied_batch[rand_idx]
            elif self.anchor_type == "debug":
                anchors = torch.zeros_like(batch.x)
            else:
                anchors = self.anchor_dist.sample([batch.x.shape[0]])
            anchor_list.append(anchors)
        anchors = torch.cat(anchor_list, dim=0)
        return anchors.to(DEVICE)

    def process_batch(self, batch, anchors):
        """
        anchors (default=None):
            if passed, will use the same set of anchors for all batches during training.
            if  None, we will randomly draw samples from the anchor dist.
        """
        if anchors == None:
            if self.anchor_type == "node":
                # anchors: [num_nodes, num_feats]
                anchors = self.anchor_dist.sample([batch.x.shape[0]]).to(DEVICE)
                new_feats = torch.cat((batch.x - anchors, anchors), dim=1)
            elif self.anchor_type == "shuffle":
                copied_batch = batch.x.clone().detach()
                rand_idx = torch.randperm(batch.x.shape[0])
                anchors = copied_batch[rand_idx]
                new_feats = torch.cat((batch.x - anchors, anchors), dim=1)
            elif self.anchor_type == "batch":
                anchors = self.anchor_dist.sample([1]).to(DEVICE)
                new_feats = torch.cat(
                    (batch.x - anchors, anchors.repeat(batch.x.shape[0], 1)), dim=1
                )
            elif self.anchor_type == "debug":
                anchors = torch.zeros_like(batch.x).to(DEVICE)
                new_feats = torch.cat((batch.x - anchors, anchors), dim=1)
            elif self.anchor_type == "graph":
                # get nodes in each graph
                counts = batch.batch.unique(return_counts=True)[1]
                anchors = self.anchor_dist.sample([batch.num_graphs]).to(DEVICE)
                anchors = torch.repeat_interleave(anchors, counts, dim=0).to(
                    DEVICE
                )  # repeat anchor to match number of nodes in each graph
                new_feats = torch.cat((batch.x - anchors, anchors), dim=1)
            elif self.anchor_type == "graph2graph":
                # assumes the same number of nodes
                # this means we can stack the anchor to match, instead of first converting to a datalist
                batch_list = batch.to_data_list()
                batch_order = np.arange(batch.num_graphs)
                np.random.shuffle(batch_order)
                anchors = torch.vstack([batch_list[i].x for i in batch_order])
                new_feats = torch.cat((batch.x - anchors, anchors), dim=1)
        else:
            """
            This is used for inference time.
            And is "node" based anchoring b/c
            we need the same anchor across different sized graphs.

            graph2graph assumes same sized graphs.
            """
            if self.anchor_type == "graph2graph":
                anchors = torch.repeat_interleave(
                    anchors.squeeze(0), batch.num_graphs, dim=0
                ).to(DEVICE)
                new_feats = torch.cat((batch.x - anchors, anchors), dim=1)
            elif self.anchor_type == "debug":
                anchors = torch.zeros_like(batch.x).to(DEVICE)
                new_feats = torch.cat((batch.x - anchors, anchors), dim=1)
            elif self.anchor_type == "shuffle":
                copied_batch = batch.x.clone().detach()
                rand_idx = torch.randperm(batch.x.shape[0])
                anchors = copied_batch[rand_idx]
                new_feats = torch.cat((batch.x - anchors, anchors), dim=1)
            else:
                anchors = torch.repeat_interleave(anchors, batch.x.shape[0], dim=0).to(
                    DEVICE
                )
                new_feats = torch.cat((batch.x - anchors, anchors), dim=1)

        return new_feats.to(DEVICE)

    def calibrate(self, mu, sig):
        """
        For ImageNet we use mu_hat = mu/c
        For CIFAR10/100 we use mu_hat = mu/(1+exp(c))
        """
        c = torch.mean(
            sig, dim=1, keepdim=True
        )  # batch-size,num-classes => batch-size,1
        c = c.expand(mu.shape)  # batch-size,1 =>
        return torch.div(mu, 1 + torch.exp(c))
        # return torch.div(mu,c)

    def forward(self, x, anchors=None, n_anchors=1, return_std=False, edge_weight=None):
        if n_anchors == 1 and return_std:
            raise Warning("Use n_anchor>1, std. dev cannot be computed!")

        preds = torch.zeros(n_anchors, x.x.shape[0], self.num_classes).to(DEVICE)
        for n in range(n_anchors):
            with torch.no_grad():
                if anchors == None:
                    new_feats = self.process_batch(x, anchors=None)
                else:
                    new_feats = self.process_batch(x, anchors=anchors[n].unsqueeze(0))
            preds[n, :, :] = self.net.forward_graph(
                new_feats, x.edge_index, edge_weight=edge_weight
            )
        # pdb.set_trace()
        mu = preds.mean(dim=0)
        if return_std:
            std = preds.sigmoid().std(dim=0)
            return self.calibrate(mu, std), std
        else:
            return mu
        

class GraphANTLayerwiseGIN(torch.nn.Module):
    def __init__(self, base_network, mean, std, anchor_type="graph", num_classes=10):
        super(GraphANTLayerwiseGIN, self).__init__()
        """
            base_network (default: None):
                network used to perform anchor training.
                first GNN layer should be modified to accept 2x the node features.     
            """
        if base_network is not None:
            self.net = base_network
        else:
            raise Exception("base network needs to be defined")

        """
        We will create hidden-representation space graph anchors.
        They can randomly be defined or defined through other samples 
        in the batch (default behavior). 
        """
        assert anchor_type.lower() in ["random", "batch"]
        self.anchor_type = anchor_type.lower()
        self.num_classes = num_classes
        self.hidden_rep_dim = 300

    """
    This function draws a fixed set of base anchors 
    for inference. Since we are performing anchoring 
    at the hidden representation space, we will return 
    [num_anchors,hidden-rep-dim].
    Process batch will have to subtract this value from each hidden rep accordingly.
    """

    def update_anchor_dist(self,dataloader,num_nodes,x_dim):

        x = torch.zeros(num_nodes, x_dim) 
        with torch.no_grad():
            start_idx = end_idx = 0
            for d in dataloader:
                d = d.to(DEVICE) 
                end_idx = start_idx + d.x.shape[0]
                x[start_idx:end_idx,:] = self.net.forward_features_pre(d).cpu()
                start_idx = end_idx
        mu = x.mean(dim=0)
        std = x.std(dim=0)
        std_mean = std.mean()
        std[std == 0] = std_mean
        
        self.mean = x.mean(dim=0)
        self.std = std
        print("Updated Mu: ", mu.shape)
        print("Updated Std: ", std.min(), std.max(), std_mean)
        assert std_mean > 0
        self.anchor_dist = torch.distributions.normal.Normal(
            loc=self.mean, scale=self.std
        )
    def get_anchors(self, batch, num_anchors=1):
        with torch.no_grad():
            if self.anchor_type == "batch":
                anchors = self.net.forward_features_pre(batch)[0:num_anchors, :]
            elif self.anchor_type == "random":
                anchors = self.anchor_dist.sample([num_anchors])
            else:
                print("INVALID ANCHOR TYPE")
                exit()
        return anchors.to(DEVICE)

    def process_batch(self, hidden_rep, anchors):
        """
        hidden_rep [num_graphs, hidden_rep_dim]: anchoring occurs before the classifier,
        so we pass the hidden rep here (not the original batch). [num_graphs, hidden_rep_dim]
        batch: we pass the original data batch too. This is needed so that we can pull out the anchors
        anchors (default=None):
            if an anchor (1) is passed, then samples in the batch will share this anchor. (inference default)
            if None, we will create a separate anchor per sample. (training default)
        """
        if anchors == None:
            if self.anchor_type == "random":
                # anchors: [num_graphs, hidden_rep_dim]
                anchors = self.anchor_dist.sample([hidden_rep.shape[0]]).to(DEVICE)
                anchors.requires_grad = False
                new_hidden_rep = torch.cat((hidden_rep - anchors, anchors), dim=1)
            elif self.anchor_type == "batch":
                # returns a cloned/detached version of the anchors
                # that are shuffled.
                anchors = hidden_rep.clone().detach()[
                    torch.randperm(hidden_rep.shape[0])
                ]
                anchors.requires_grad = False                
                new_hidden_rep = torch.cat((hidden_rep - anchors, anchors), dim=1)
        else:
            anchors = torch.repeat_interleave(anchors, hidden_rep.shape[0], dim=0).to(
                DEVICE
            )
            anchors.requires_grad = False
            new_hidden_rep = torch.cat((hidden_rep - anchors, anchors), dim=1)

        return new_hidden_rep.to(DEVICE)

    def calibrate(self, mu, sig):
        """
        For ImageNet we use mu_hat = mu/c
        For CIFAR10/100 we use mu_hat = mu/(1+exp(c))
        """
        c = torch.mean(
            sig, dim=1, keepdim=True
        )  # batch-size,num-classes => batch-size,1
        c = c.expand(mu.shape)  # batch-size,1 =>
        return torch.div(mu, 1 + torch.exp(c))
        # return torch.div(mu,c)

    """
    This computation graph is more 
    complicated than the original delta-UQ formulation.
    Gradients will be propagated through the residual operation.
    However, they should not be propagated through the hidden anchor representation
    (so we are not computing the gradient twice).
    """

    def forward(self, batch, anchors=None, n_anchors=1, return_std=False,return_unq=False):
        if n_anchors == 1 and return_std:
            raise Warning("Use n_anchor>1, std. dev cannot be computed!")

        preds = torch.zeros(n_anchors, batch.num_graphs, self.num_classes).to(DEVICE)
        clean_pre_reps = self.net.forward_features_pre(batch)
        for n in range(n_anchors):
            if anchors == None:
                new_feats = self.process_batch(clean_pre_reps, anchors=None)
            else:
                new_feats = self.process_batch(
                    clean_pre_reps, anchors=anchors[n].unsqueeze(0)
                )

                # new_feats = self.process_batch(
                #     clean_pre_reps, anchors=anchors[n].unsqueeze(0)
                # )
            preds[n, :, :] = self.net.forward_classifier(self.net.forward_features_post(new_feats,batch.edge_index, batch.batch))
        mu = preds.mean(dim=0)
        if return_std:
            std = preds.sigmoid().std(dim=0)
            # std = preds.std(dim=0)
            return self.calibrate(mu, std), std
        else:
            return mu

class GraphANTLayerwisevGIN(torch.nn.Module):
    def __init__(self, base_network, mean, std, anchor_type="graph", num_classes=10):
        super(GraphANTLayerwisevGIN, self).__init__()
        """
            base_network (default: None):
                network used to perform anchor training.
                first GNN layer should be modified to accept 2x the node features.     
            """
        if base_network is not None:
            self.net = base_network
        else:
            raise Exception("base network needs to be defined")

        """
        We will create hidden-representation space graph anchors.
        They can randomly be defined or defined through other samples 
        in the batch (default behavior). 
        """
        assert anchor_type.lower() in ["random", "batch"]
        self.anchor_type = anchor_type.lower()
        self.num_classes = num_classes
        self.hidden_rep_dim = 300

    """
    This function draws a fixed set of base anchors 
    for inference. Since we are performing anchoring 
    at the hidden representation space, we will return 
    [num_anchors,hidden-rep-dim].
    Process batch will have to subtract this value from each hidden rep accordingly.
    """

    def update_anchor_dist(self,dataloader,num_nodes,x_dim):

        x = torch.zeros(num_nodes, x_dim) 
        with torch.no_grad():
            start_idx = end_idx = 0
            for d in dataloader:
                d = d.to(DEVICE) 
                end_idx = start_idx + d.x.shape[0]
                feat, v_node_feat = self.net.forward_features_pre(d)
                x[start_idx:end_idx,:] = feat.cpu()
                start_idx = end_idx
        mu = x.mean(dim=0)
        std = x.std(dim=0)
        std_mean = std.mean()
        std[std == 0] = std_mean
        
        self.mean = x.mean(dim=0)
        self.std = std
        print("Updated Mu: ", mu.shape)
        print("Updated Std: ", std.min(), std.max(), std_mean)
        assert std_mean > 0
        self.anchor_dist = torch.distributions.normal.Normal(
            loc=self.mean, scale=self.std
        )

    def get_anchors(self, batch, num_anchors=1):
        with torch.no_grad():
            if self.anchor_type == "batch":
                feat, v_node_feat = self.net.forward_features_pre(batch)
                anchors = feat[0:num_anchors, :]
            elif self.anchor_type == "random":
                anchors = self.anchor_dist.sample([num_anchors])
            else:
                print("INVALID ANCHOR TYPE")
                exit()
        return anchors.to(DEVICE)

    def process_batch(self, hidden_rep, anchors):
        """
        hidden_rep [num_graphs, hidden_rep_dim]: anchoring occurs before the classifier,
        so we pass the hidden rep here (not the original batch). [num_graphs, hidden_rep_dim]
        batch: we pass the original data batch too. This is needed so that we can pull out the anchors
        anchors (default=None):
            if an anchor (1) is passed, then samples in the batch will share this anchor. (inference default)
            if None, we will create a separate anchor per sample. (training default)
        """
        if anchors == None:
            if self.anchor_type == "random":
                # anchors: [num_graphs, hidden_rep_dim]
                anchors = self.anchor_dist.sample([hidden_rep.shape[0]]).to(DEVICE)
                anchors.requires_grad = False
                new_hidden_rep = torch.cat((hidden_rep - anchors, anchors), dim=1)
            elif self.anchor_type == "batch":
                # returns a cloned/detached version of the anchors
                # that are shuffled.
                anchors = hidden_rep.clone().detach()[
                    torch.randperm(hidden_rep.shape[0])
                ]
                anchors.requires_grad = False                
                new_hidden_rep = torch.cat((hidden_rep - anchors, anchors), dim=1)
        else:
            anchors = torch.repeat_interleave(anchors, hidden_rep.shape[0], dim=0).to(
                DEVICE
            )
            anchors.requires_grad = False
            new_hidden_rep = torch.cat((hidden_rep - anchors, anchors), dim=1)

        return new_hidden_rep.to(DEVICE)

    def calibrate(self, mu, sig):
        """
        For ImageNet we use mu_hat = mu/c
        For CIFAR10/100 we use mu_hat = mu/(1+exp(c))
        """
        c = torch.mean(
            sig, dim=1, keepdim=True
        )  # batch-size,num-classes => batch-size,1
        c = c.expand(mu.shape)  # batch-size,1 =>
        return torch.div(mu, 1 + torch.exp(c))
        # return torch.div(mu,c)

    """
    This computation graph is more 
    complicated than the original delta-UQ formulation.
    Gradients will be propagated through the residual operation.
    However, they should not be propagated through the hidden anchor representation
    (so we are not computing the gradient twice).
    """

    def forward(self, batch, anchors=None, n_anchors=1, return_std=False,return_unq=False):
        if n_anchors == 1 and return_std:
            raise Warning("Use n_anchor>1, std. dev cannot be computed!")

        preds = torch.zeros(n_anchors, batch.num_graphs, self.num_classes).to(DEVICE)
        clean_pre_reps, v_node_feat = self.net.forward_features_pre(batch)
        for n in range(n_anchors):
            if anchors == None:
                new_feats = self.process_batch(clean_pre_reps, anchors=None)
            else:
                new_feats = self.process_batch(
                    clean_pre_reps, anchors=anchors[n].unsqueeze(0)
                )

            preds[n, :, :] = self.net.forward_classifier(self.net.forward_features_post(v_node_feat,new_feats,batch.edge_index, batch.batch))
        mu = preds.mean(dim=0)
        if return_std:
            std = preds.sigmoid().std(dim=0)
            # std = preds.std(dim=0)
            return self.calibrate(mu, std), std
        else:
            return mu
        

def test_node(model,loader,acc_metric,config, anchors=None,split='val'):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad(): 
        for batch in tqdm.tqdm(loader,disable=True):
            batch=batch.to(DEVICE)
            mask, labels= nan2zero_get_mask(batch, split, config)
            out = model(batch,anchors=anchors,n_anchors=config.uq.num_anchors) #FIXME!
            out = out[mask]
            preds.append(out.to('cpu'))
            targets.append(labels[mask].to('cpu'))
    preds = torch.cat(preds,dim=0)
    preds =  preds.argmax(dim=1)
    targets = torch.cat(targets,dim=0)
    acc = acc_metric(preds,targets.reshape(-1))
    return acc


class baseModelNode(torch.nn.Module):
    def __init__(self,encoder,num_classes=3):
        super().__init__()
        self.encoder = encoder
        self.classifier = torch.nn.Sequential(torch.nn.Linear(300,num_classes)) 
        self.num_classes = num_classes 
    def forward(self, d_1):
        x_1 = self.encoder(x= d_1.x, edge_index=d_1.edge_index,edge_weight=None, batch=None)
        output = self.classifier(x_1)
        return output
    def forward_graph(self, x,edge_index,edge_weight=None,batch=None):
        x_1 = self.encoder(x=x, edge_index=edge_index,edge_weight=edge_weight,batch=batch)
        output = self.classifier(x_1)
        return output

class GraphANTNode(torch.nn.Module):
    def __init__(self,base_network,mean,std,anchor_type="graph",num_classes=10):
            super(GraphANTNode, self).__init__()
            '''
            base_network (default: None):
                network used to perform anchor training.
                first GNN layer should be modified to accept 2x the node features.     
            '''
            if base_network is not None:
                self.net = base_network
            else:
                raise Exception('base network needs to be defined')

            self.mean = mean
            self.std = std
            """
            we will sample anchors from this distribution. Anchors can be: 
            -- NODE-specific (different anchor per node)
            -- GRAPH-specific (different anchor per graph) 
            """
            assert anchor_type.lower() in ['node','graph','batch','graph2graph','shuffle','debug']
            self.anchor_type = anchor_type.lower()
            self.anchor_dist = torch.distributions.normal.Normal(loc=self.mean, scale=self.std)
            self.num_classes = num_classes

    """
    This function draws a fixed set of base anchors 
    for inference. At inference time, we must use "node" anchoring.
    Therefore, we return [num_anchors,node_features].
    Process batch will have to subtract this value from each node accordingly.
    """
    def get_anchors(self,batch,num_anchors=1):
        anchor_list = []
        if self.anchor_type == 'graph2graph':
            batch = batch.to_data_list()
        for i in range(num_anchors):
            if self.anchor_type == 'graph2graph':
                anchors = batch[i].x.unsqueeze(0) #select anchors from batch, assumes the same number of nodes!! 
            elif self.anchor_type == 'shuffle':
                copied_batch = batch.x.clone().detach()
                #shuffle the copied  batch along x
                rand_idx = torch.randperm(batch.x.shape[0])
                anchors = copied_batch[rand_idx]
            elif self.anchor_type == 'debug':
                anchors = torch.zeros_like(batch.x)
            else:
                anchors = self.anchor_dist.sample([batch.x.shape[0]])
            anchor_list.append(anchors)
        anchors = torch.cat(anchor_list,dim=0)
        return anchors.to(DEVICE) 
    
    def process_batch(self,batch,anchors):
        '''
        anchors (default=None):
            if passed, will use the same set of anchors for all batches during training.
            if  None, we will randomly draw samples from the anchor dist.
        '''
        if anchors == None:
            if self.anchor_type == 'node':
                #anchors: [num_nodes, num_feats]
                anchors = self.anchor_dist.sample([batch.x.shape[0]]).to(DEVICE)
                new_feats = torch.cat((batch.x - anchors,anchors),dim=1)
            elif self.anchor_type == 'shuffle':
                copied_batch = batch.x.clone().detach()
                rand_idx = torch.randperm(batch.x.shape[0])
                anchors = copied_batch[rand_idx]
                new_feats = torch.cat((batch.x - anchors,anchors),dim=1)
            elif self.anchor_type == 'batch':
                anchors = self.anchor_dist.sample([1]).to(DEVICE)
                new_feats = torch.cat((batch.x - anchors,anchors.repeat(batch.x.shape[0],1)),dim=1)
            elif self.anchor_type == 'debug':
                anchors = torch.zeros_like(batch.x).to(DEVICE)
                new_feats = torch.cat((batch.x - anchors,anchors),dim=1)
            elif self.anchor_type == 'graph':
                #get nodes in each graph 
                counts = batch.batch.unique(return_counts=True)[1]
                anchors = self.anchor_dist.sample([batch.num_graphs]).to(DEVICE)
                anchors = torch.repeat_interleave(anchors, counts, dim=0).to(DEVICE) #repeat anchor to match number of nodes in each graph
                new_feats = torch.cat((batch.x - anchors,anchors),dim=1)
            elif self.anchor_type == 'graph2graph':
                #assumes the same number of nodes
                #this means we can stack the anchor to match, instead of first converting to a datalist  
                batch_list = batch.to_data_list()
                batch_order = np.arange(batch.num_graphs) 
                np.random.shuffle(batch_order)
                anchors = torch.vstack([batch_list[i].x for i in batch_order] )
                new_feats = torch.cat((batch.x - anchors,anchors),dim=1)
        else:
            """
            This is used for inference time.
            And is "node" based anchoring b/c
            we need the same anchor across different sized graphs.

            graph2graph assumes same sized graphs.
            """
            if self.anchor_type == 'graph2graph':
                anchors = torch.repeat_interleave(anchors.squeeze(0), batch.num_graphs, dim=0).to(DEVICE) 
                new_feats = torch.cat((batch.x - anchors,anchors),dim=1)
            elif self.anchor_type == 'debug':
                anchors = torch.zeros_like(batch.x).to(DEVICE)
                new_feats = torch.cat((batch.x - anchors,anchors),dim=1)
            elif self.anchor_type == 'shuffle':
                copied_batch = batch.x.clone().detach()
                rand_idx = torch.randperm(batch.x.shape[0])
                anchors = copied_batch[rand_idx]
                new_feats = torch.cat((batch.x - anchors,anchors),dim=1)
            else:
                anchors = torch.repeat_interleave(anchors, batch.x.shape[0], dim=0).to(DEVICE) 
                new_feats = torch.cat((batch.x - anchors,anchors),dim=1)

        return new_feats.to(DEVICE) 

    def calibrate(self,mu,sig):
        '''
        For ImageNet we use mu_hat = mu/c
        For CIFAR10/100 we use mu_hat = mu/(1+exp(c))
        '''
        c = torch.mean(sig,dim=1,keepdim=True) #batch-size,num-classes => batch-size,1
        c = c.expand(mu.shape) #batch-size,1 => 
        return torch.div(mu,1+torch.exp(c))
        #return torch.div(mu,c)

    def forward(self,x,anchors=None,n_anchors=1,return_std=False,edge_weight=None):
        if n_anchors==1 and return_std:
            raise Warning('Use n_anchor>1, std. dev cannot be computed!')

        preds = torch.zeros(n_anchors,x.x.shape[0],self.num_classes).to(DEVICE) 
        for n in range(n_anchors):
            with torch.no_grad():
                if anchors == None:
                    new_feats = self.process_batch(x,anchors=None)
                else:
                    new_feats = self.process_batch(x,anchors=anchors[n].unsqueeze(0))
            preds[n,:,:] = self.net.forward_graph(x=new_feats,edge_index=x.edge_index,edge_weight=edge_weight)
        # pdb.set_trace()
        mu = preds.mean(dim=0)
        if return_std:
            std = preds.sigmoid().std(dim=0)
            return self.calibrate(mu,std), std
        else:
            return mu

