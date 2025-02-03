import os
import shutil
import numpy as np
import torch
from torch_geometric.data import Data

def load_labels(label_paths):
    labels = []
    for label_path in label_paths:
        with open(label_path, "r") as f:
            class_ids = [int(line.split()[0]) for line in f.readlines()]
        labels.append(max(set(class_ids)) if class_ids else 0)
    return np.array(labels)

def save_pseudo_labels(pseudo_labels, confidence_mask, unlabeled_image_paths, output_dir, model):
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, (img_path, label, is_confident) in enumerate(zip(
            unlabeled_image_paths, 
            pseudo_labels[len(unlabeled_image_paths):], 
            confidence_mask[len(unlabeled_image_paths):])):
        
        if is_confident:
            img_name = os.path.basename(img_path).rsplit('.', 1)[0]
            label_path = os.path.join(output_dir, f"{img_name}.txt")
            results = model(img_path)
            
            with open(label_path, "w") as f:
                for result in results:
                    for box in result.boxes:
                        if box.conf > 0.6:
                            x_center, y_center, width, height = box.xywhn.cpu().numpy().flatten()
                            f.write(f"{int(label.item())} {x_center} {y_center} {width} {height}\n")

def prepare_graph_data(labeled_features, unlabeled_features, edge_index, labels):
    return Data(
        x=torch.tensor(np.vstack([labeled_features, unlabeled_features]), dtype=torch.float32),
        edge_index=edge_index,
        y=torch.cat([torch.tensor(labels, dtype=torch.long), 
                    torch.full((len(unlabeled_features),), -1)])
    )
