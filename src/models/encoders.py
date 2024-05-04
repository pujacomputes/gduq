import pdb
import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, GCNConv  
from torch_geometric.nn.pool import global_add_pool as GlobalAddPool
from torch_geometric.nn.pool import global_mean_pool as GlobalMeanPool 
from torch_geometric.nn.pool import global_max_pool as GlobalMaxPool
"""
While we directly modified the GOOD package to register our models,
we've included the code for the encoders here for reference here.
The goal is to allow users to download the GOOD package without other modifications.
"""

class GINEncoder(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        num_layer = config.model.model_layer
        self.conv1 = GINConv(nn.Sequential(nn.Linear(config.dataset.dim_node, 2 * config.model.dim_hidden),
                                           nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                                           nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))
        self.convs = nn.ModuleList(
            [
                GINConv(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                      nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                                      nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))
                for _ in range(num_layer - 1)
            ]
        )

        num_layer = config.model.model_layer

        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList(
            [
                nn.ReLU()
                for _ in range(num_layer - 1)
            ]
        )
        self.batch_norm1 = nn.BatchNorm1d(config.model.dim_hidden)
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(config.model.dim_hidden)
            for _ in range(num_layer - 1)
        ])
        self.dropout1 = nn.Dropout(config.model.dropout_rate)
        self.dropouts = nn.ModuleList([
            nn.Dropout(config.model.dropout_rate)
            for _ in range(num_layer - 1)
        ])
        if config.model.model_level == 'node':
            self.readout =torch.nn.Identity 
        elif config.model.global_pool == 'mean':
            self.readout = GlobalMeanPool
        else:
            self.readout = GlobalMaxPool



    def forward(self, x, edge_index, batch):
        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            post_conv = batch_norm(conv(post_conv, edge_index))
            if i != len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)

        out_readout = self.readout(post_conv, batch)
        return out_readout

class GCNEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        num_layer = config.model.model_layer

        self.conv1 = GCNConv(config.dataset.dim_node, config.model.dim_hidden)
        self.convs = nn.ModuleList(
            [
                GCNConv(config.model.dim_hidden, config.model.dim_hidden)
                for _ in range(num_layer - 1)
            ]
        )
        num_layer = config.model.model_layer

        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList(
            [
                nn.ReLU()
                for _ in range(num_layer - 1)
            ]
        )
        self.batch_norm1 = nn.BatchNorm1d(config.model.dim_hidden)
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(config.model.dim_hidden)
            for _ in range(num_layer - 1)
        ])
        self.dropout1 = nn.Dropout(config.model.dropout_rate)
        self.dropouts = nn.ModuleList([
            nn.Dropout(config.model.dropout_rate)
            for _ in range(num_layer - 1)
        ])
        self.model_level = config.model.model_level
        if config.model.model_level == 'node':
            self.readout =torch.nn.Identity() 
        elif config.model.global_pool == 'mean':
            self.readout = GlobalMeanPool
        else:
            self.readout = GlobalMaxPool
    
    def forward(self, x, edge_index, edge_weight, batch):
        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index, edge_weight))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            post_conv = batch_norm(conv(post_conv, edge_index, edge_weight))
            if i < len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)

        if self.model_level == 'node':
            out_readout = post_conv
        else:
            out_readout = self.readout(post_conv, batch)
        return out_readout
    


class VirtualNodeEncoder(torch.nn.Module):
   
    def __init__(self, config):
        super().__init__()
        self.virtual_node_embedding = nn.Embedding(1, config.model.dim_hidden)
        self.virtual_mlp = nn.Sequential(*(
                [nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                 nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU()] +
                [nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden),
                 nn.BatchNorm1d(config.model.dim_hidden), nn.ReLU(),
                 nn.Dropout(config.model.dropout_rate)]
        ))
        self.virtual_pool = GlobalAddPool

    def forward(self,x):

        return 
class vGINEncoder(GINEncoder):
    r"""
    The vGIN encoder for non-molecule data, using the :class:`~vGINConv` operator for message passing.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.model.dropout_rate`)
    """

    def __init__(self, config):
        super().__init__(config)
        self.virtualnodeencoder = VirtualNodeEncoder(config)
        self.virtual_node_embedding = self.virtualnodeencoder.virtual_node_embedding
        self.virtual_mlp = self.virtualnodeencoder.virtual_mlp
        self.virtual_pool = self.virtualnodeencoder.virtual_pool
        self.config = config

    def forward(self, x, edge_index, batch):
        virtual_node_feat = self.virtual_node_embedding(
            torch.zeros(batch[-1].item() + 1, device=self.config.device, dtype=torch.long))

        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            # --- Add global info ---
            post_conv = post_conv + virtual_node_feat[batch]
            post_conv = batch_norm(conv(post_conv, edge_index))
            if i < len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)
            # --- update global info ---
            if i < len(self.convs) - 1:
                virtual_node_feat = self.virtual_mlp(self.virtual_pool(post_conv, batch) + virtual_node_feat)

        out_readout = self.readout(post_conv, batch)
        return out_readout



class GCNEncoderLayerwise(torch.nn.Module):
    def __init__(self, config):
        super(GCNEncoderLayerwise, self).__init__(config)
        num_layer = config.model.model_layer
        self.layerwise_duq = config.uq.layerwise_duq
        
        self.conv1 = GCNConv(config.dataset.dim_node, config.model.dim_hidden)
        self.convs = nn.ModuleList()
        for i in range(1, num_layer):
            if i == self.layerwise_duq:
                in_dim = 2 * config.model.dim_hidden #[x, x-c]
                out_dim = config.model.dim_hidden 
            else: 
                in_dim = config.model.dim_hidden
                out_dim = config.model.dim_hidden 
            self.convs.append(GCNConv(in_dim, out_dim))
        self.convs.insert(0,self.conv1) 
        self.batch_norms.insert(0,self.batch_norm1)
        self.relus.insert(0,self.relu1)
        self.dropouts.insert(0,self.dropout1)

    def forward(self, x, edge_index, edge_weight, batch):
        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index, edge_weight))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            post_conv = batch_norm(conv(post_conv, edge_index, edge_weight))
            if i < len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)

        out_readout = self.readout(post_conv, batch)
        return out_readout
    def forward_post(self,  x, edge_index, edge_weight, batch):
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs[self.layerwise_duq:], self.batch_norms[self.layerwise_duq:], self.relus[self.layerwise_duq:], self.dropouts[self.layerwise_duq:])):
            x = batch_norm(conv(x, edge_index,edge_weight))
            i = i + self.layerwise_duq
            if i != len(self.convs) - 1:
                x = relu(x)
            x = dropout(x)

        out_readout = self.readout(x, batch)
        return out_readout
    
    
    def forward_pre(self, x, edge_index, edge_weight, batch):
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs[:self.layerwise_duq], self.batch_norms[:self.layerwise_duq], self.relus[:self.layerwise_duq], self.dropouts[:self.layerwise_duq])):
            x = batch_norm(conv(x,edge_index,edge_weight))
            if i != len(self.convs) - 1:
                x = relu(x)
            x = dropout(x)
        return x 


class GINEncoderLayerwise(torch.nn.Module):
    def __init__(self, config):

        super(GINEncoderLayerwise, self).__init__(config)
        num_layer = config.model.model_layer
        self.layerwise_duq = config.uq.layerwise_duq
        self.conv1 = GINConv(nn.Sequential(nn.Linear(config.dataset.dim_node, 2 * config.model.dim_hidden),
                                           nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                                           nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))

        self.convs = torch.nn.ModuleList()
        for i in range(1, num_layer):
            if i == self.layerwise_duq:
                in_dim = 2 * config.model.dim_hidden #[x, x-c]
                inter_dim = 2 * in_dim 
                out_dim = config.model.dim_hidden 
            else: 
                in_dim = config.model.dim_hidden
                inter_dim = 2 * in_dim 
                out_dim = in_dim
            self.convs.append(GINConv(nn.Sequential(
                                    nn.Linear(in_dim, inter_dim),
                                    nn.BatchNorm1d(inter_dim), nn.ReLU(),
                                    nn.Linear(inter_dim, out_dim))))
        
        self.convs.insert(0,self.conv1)
        self.batch_norms.insert(0,self.batch_norm1)
        self.relus.insert(0,self.relu1)
        self.dropouts.insert(0,self.dropout1)
    
    def forward_post(self, x, edge_index, batch):
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs[self.layerwise_duq:], self.batch_norms[self.layerwise_duq:], self.relus[self.layerwise_duq:], self.dropouts[self.layerwise_duq:])):
            x = batch_norm(conv(x, edge_index))
            i = i + self.layerwise_duq
            if i != len(self.convs) - 1:
                x = relu(x)
            x = dropout(x)

        out_readout = self.readout(x, batch)
        return out_readout
    
    def forward_pre(self, x, edge_index, batch):
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs[:self.layerwise_duq], self.batch_norms[:self.layerwise_duq], self.relus[:self.layerwise_duq], self.dropouts[:self.layerwise_duq])):
            x = batch_norm(conv(x,edge_index))
            if i != len(self.convs) - 1:
                x = relu(x)
            x = dropout(x)
        return x 
    
    def forward(self, x, edge_index, batch):
        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs[1:], self.batch_norms[1:], self.relus[1:], self.dropouts[1:])):
            post_conv = batch_norm(conv(post_conv, edge_index))
            if i != len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)

        out_readout = self.readout(post_conv, batch)
        return out_readout


class vGINEncoderLayerwise(GINEncoder):

    def __init__(self, config):
        super(vGINEncoderLayerwise, self).__init__(config)
        self.config = config

    def forward(self, x, edge_index, batch):
        virtual_node_feat = self.virtual_node_embedding(
            torch.zeros(batch[-1].item() + 1, device=self.config.device, dtype=torch.long))

        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            # --- Add global info ---
            post_conv = post_conv + virtual_node_feat[batch]
            post_conv = batch_norm(conv(post_conv, edge_index))
            if i < len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)
            # --- update global info ---
            if i < len(self.convs) - 1:
                virtual_node_feat = self.virtual_mlp(self.virtual_pool(post_conv, batch) + virtual_node_feat)

        out_readout = self.readout(post_conv, batch)
        return out_readout
    def forward_post(self, virtual_node_feat, x, edge_index, batch):

        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs[self.layerwise_duq:], self.batch_norms[self.layerwise_duq:], self.relus[self.layerwise_duq:], self.dropouts[self.layerwise_duq:])):
            # --- Add global info ---
            if i == 0:
                x = x + virtual_node_feat[batch].repeat(1,2)
            else:
                x = x + virtual_node_feat[batch]
            x = batch_norm(conv(x, edge_index))
            if i < len(self.convs) - 1:
                x = relu(x)
            x = dropout(x)
            # --- update global info ---
            if i < len(self.convs) - 1:
                virtual_node_feat = self.virtual_mlp(self.virtual_pool(x, batch) + virtual_node_feat)
        out_readout = self.readout(x, batch)
        return out_readout
    
    def forward_pre(self, x, edge_index, batch):
        virtual_node_feat = self.virtual_node_embedding(
            torch.zeros(batch[-1].item() + 1, device=self.config.device, dtype=torch.long))
        x = self.dropouts[0](self.relus[0](self.batch_norms[0](self.convs[0](x, edge_index))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs[1:self.layerwise_duq], self.batch_norms[1:self.layerwise_duq], self.relus[1:self.layerwise_duq], self.dropouts[1:self.layerwise_duq])):
            # --- Add global info ---
            x = x + virtual_node_feat[batch]
            x = batch_norm(conv(x, edge_index))
            if i < len(self.convs) - 1:
                x = relu(x)
            x = dropout(x)
            # --- update global info ---
            if i < len(self.convs) - 1:
                virtual_node_feat = self.virtual_mlp(self.virtual_pool(x, batch) + virtual_node_feat)
        return x, virtual_node_feat