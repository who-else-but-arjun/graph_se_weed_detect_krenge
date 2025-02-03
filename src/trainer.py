import torch
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops
import numpy as np

def calculate_loss(out, graph_data, pseudo_labels, pseudo_mask, model):
    labeled_mask = graph_data.y != -1
    supervised_loss = F.nll_loss(out[labeled_mask], graph_data.y[labeled_mask])
    
    pseudo_loss = 0.0
    if pseudo_mask.any():
        pseudo_loss = F.nll_loss(out[pseudo_mask], pseudo_labels[pseudo_mask])
    
    edge_index, _ = remove_self_loops(graph_data.edge_index)
    smoothness_loss = torch.mean(torch.pow(
        out[edge_index[0]] - out[edge_index[1]], 2
    ))
    
    pseudo_weight = min(0.3, model.training_step / 100)
    total_loss = supervised_loss + pseudo_weight * pseudo_loss + 0.1 * smoothness_loss
    
    return total_loss, {
        'supervised': supervised_loss.item(),
        'pseudo': pseudo_loss.item() if isinstance(pseudo_loss, torch.Tensor) else pseudo_loss,
        'smoothness': smoothness_loss.item(),
        'total': total_loss.item()
    }

def generate_pseudo_labels(model, graph_data, confidence_threshold=0.9):
    model.eval()
    with torch.no_grad():
        out = model(graph_data.x, graph_data.edge_index)
        probs = torch.exp(out)
        max_probs, pseudo_labels = probs.max(dim=1)
        
        unlabeled_mask = graph_data.y == -1
        mean_confidence = max_probs[unlabeled_mask].mean()
        adaptive_threshold = min(confidence_threshold, mean_confidence + 0.1)
        
        confidence_mask = (max_probs > adaptive_threshold) & unlabeled_mask
        
        if confidence_mask.any():
            class_dist = torch.bincount(pseudo_labels[confidence_mask])
            if len(class_dist) > 1:
                ratio = float(class_dist.max()) / class_dist.min()
                if ratio > 3:
                    minor_class = class_dist.argmin()
                    confidence_mask &= ((pseudo_labels == minor_class) | 
                                     (max_probs > (adaptive_threshold + 0.1)))
        
        return pseudo_labels, confidence_mask
