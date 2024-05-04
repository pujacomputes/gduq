import torch
import numpy as np

class baseModel(torch.nn.Module):
    def __init__(self, encoder, num_classes=3,classifier_dim=300):
        super().__init__()
        self.encoder = encoder
        self.classifier = torch.nn.Sequential(torch.nn.Linear(classifier_dim, num_classes))
        self.num_classes = num_classes

    def forward(self, d_1):
        x_1 = self.encoder(x=d_1.x, edge_index=d_1.edge_index, batch=d_1.batch)
        output = self.classifier(x_1)
        return output

    def forward_graph(self, x, edge_index, batch):
        x_1 = self.encoder(x, edge_index, batch)
        output = self.classifier(x_1)
        return output

    def forward_features(self, d_1):
        x_1 = self.encoder(d_1.x, d_1.edge_index, d_1.batch)
        return x_1

    def forward_classifier(self, d_1):
        output = self.classifier(d_1)
        return output

class baseModelvGIN(torch.nn.Module):
    def __init__(self, encoder, num_classes=3,classifier_dim=300):
        super().__init__()
        self.encoder = encoder
        self.classifier = torch.nn.Sequential(torch.nn.Linear(classifier_dim, num_classes))
        self.num_classes = num_classes

    def forward_features_pre(self, d_1):
        x_1,virtual_node_feat = self.encoder.forward_pre(d_1.x, d_1.edge_index, d_1.batch)
        return x_1, virtual_node_feat
    
    def forward_features_post(self,virtual_node_feat, x, edge_index, batch):
        x_1 = self.encoder.forward_post(virtual_node_feat,x, edge_index, batch)
        return x_1
    
    def forward_classifier(self, d_1):
        output = self.classifier(d_1)
        return output


class baseModelvGINLayerwise(torch.nn.Module):
    def __init__(self, encoder, num_classes=3,classifier_dim=300):
        super().__init__()
        self.encoder = encoder
        self.classifier = torch.nn.Sequential(torch.nn.Linear(classifier_dim, num_classes))
        self.num_classes = num_classes

    def forward_features_pre(self, d_1):
        x_1,virtual_node_feat = self.encoder.forward_pre(d_1.x, d_1.edge_index, d_1.batch)
        return x_1, virtual_node_feat
    
    def forward_features_post(self,virtual_node_feat, x, edge_index, batch):
        x_1 = self.encoder.forward_post(virtual_node_feat,x, edge_index, batch)
        return x_1
    
    def forward_classifier(self, d_1):
        output = self.classifier(d_1)
        return output

class baseModelGINLayerwise(torch.nn.Module):
    def __init__(self, encoder, num_classes=3,classifier_dim=300):
        super().__init__()
        self.encoder = encoder
        self.classifier = torch.nn.Sequential(torch.nn.Linear(classifier_dim, num_classes))
        self.num_classes = num_classes

    def forward(self, d_1):
        x_1 = self.encoder(d_1.x, d_1.edge_index, d_1.batch)
        output = self.classifier(x_1)
        return output

    def forward_graph(self, x, edge_index, batch):
        x_1 = self.encoder(x, edge_index, batch)
        output = self.classifier(x_1)
        return output

    def forward_features_pre(self, d_1):
        x_1 = self.encoder.forward_pre(d_1.x, d_1.edge_index, d_1.batch)
        return x_1

    def forward_features_post(self, x, edge_index, batch):
        x_1 = self.encoder.forward_post(x, edge_index, batch)
        return x_1
        
    def forward_classifier(self, d_1):
        output = self.classifier(d_1)
        return output