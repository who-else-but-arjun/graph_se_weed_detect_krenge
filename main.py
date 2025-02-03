import argparse
import os
import shutil
import numpy as np
import torch
from ultralytics import YOLO
from src.feature_extractor import FeatureExtractor
from src.graph_constructor import construct_knn_graph
from src.gnn_model import GNN
from src.trainer import calculate_loss, generate_pseudo_labels
from src.utils import load_labels, save_pseudo_labels, prepare_graph_data

def parse_args():
    parser = argparse.ArgumentParser(description='Semi-supervised object detection training')
    parser.add_argument('--labeled_dir', required=True, help='Directory containing labeled data')
    parser.add_argument('--unlabeled_dir', required=True, help='Directory containing unlabeled data')
    parser.add_argument('--dataset_yaml', required=True, help='Path to initial dataset YAML')
    parser.add_argument('--dataset_updated_yaml', required=True, help='Path to updated dataset YAML')
    parser.add_argument('--test_yaml', required=True, help='Path to test dataset YAML')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--yolo_model', default='yolov8l.pt', help='YOLO model path')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--gnn_hidden_dim', type=int, default=128, help='GNN hidden dimension')
    parser.add_argument('--confidence_threshold', type=float, default=0.9, help='Pseudo-labeling confidence threshold')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Starting initial YOLO training...")
    model = YOLO(args.yolo_model)
    model.train(
        data=args.dataset_yaml,
        epochs=args.epochs,
        imgsz=512,
        batch=args.batch_size,
        lr0=args.learning_rate,
        lrf=0.002,
        momentum=0.98,
        weight_decay=0.0001,
        optimizer="RAdam",
        cos_lr=True,
        warmup_epochs=30,
        iou=0.5,
        mosaic=0.0,
        dfl=2.0,
        device="cuda",
        augment=True,
        verbose=True
    )
    
    print("Extracting features from images...")
    labeled_image_paths = [os.path.join(args.labeled_dir, "images", f) 
                          for f in os.listdir(os.path.join(args.labeled_dir, "images"))]
    unlabeled_image_paths = [os.path.join(args.unlabeled_dir, f) 
                            for f in os.listdir(args.unlabeled_dir) 
                            if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    feature_extractor = FeatureExtractor(model)
    labeled_features = feature_extractor.extract_features(labeled_image_paths)
    unlabeled_features = feature_extractor.extract_features(unlabeled_image_paths)
    
    print("Constructing KNN graph...")
    edge_index, edge_weights = construct_knn_graph(
        np.vstack([labeled_features, unlabeled_features])
    )
    
    labeled_label_paths = [os.path.join(args.labeled_dir, "labels", 
                          os.path.basename(p).rsplit('.', 1)[0] + '.txt') 
                          for p in labeled_image_paths]
    actual_labels = load_labels(labeled_label_paths)
    graph_data = prepare_graph_data(labeled_features, unlabeled_features, 
                                  edge_index, actual_labels)
    
    print("Training GNN model...")
    gnn_model = GNN(in_dim=graph_data.x.shape[1], 
                    hidden_dim=args.gnn_hidden_dim)
    optimizer = torch.optim.AdamW(gnn_model.parameters(), 
                                lr=0.001, 
                                weight_decay=0.01, 
                                amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5
    )
    
    best_loss = float('inf')
    for epoch in range(args.epochs):
        gnn_model.train()
        optimizer.zero_grad()
        out = gnn_model(graph_data.x, graph_data.edge_index)
        
        pseudo_labels, confidence_mask = generate_pseudo_labels(
            gnn_model, graph_data, args.confidence_threshold
        )
        loss, loss_components = calculate_loss(
            out, graph_data, pseudo_labels, confidence_mask, gnn_model
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss_components['total'])
        
        gnn_model.training_step += 1
        
        if loss_components['total'] < best_loss:
            best_loss = loss_components['total']
            torch.save(gnn_model.state_dict(), 
                      os.path.join(args.output_dir, 'gnn_model.pt'))
        
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch}:")
            for k, v in loss_components.items():
                print(f"{k.capitalize()} Loss: {v:.4f}")
            print(f"Confident Pseudo Labels: {confidence_mask.sum().item()}")
            if confidence_mask.any():
                print(f"Class distribution: {torch.bincount(pseudo_labels[confidence_mask])}")
    
    print("Generating pseudo labels...")
    gnn_model.load_state_dict(torch.load(os.path.join(args.output_dir, 'gnn_model.pt')))
    gnn_model.eval()
    
    with torch.no_grad():
        final_out = gnn_model(graph_data.x, graph_data.edge_index)
        pseudo_labels = final_out.argmax(dim=1)
        confidence_mask = torch.exp(final_out).max(dim=1)[0] > args.confidence_threshold
    
    pseudo_label_dir = os.path.join(args.output_dir, 'pseudo_labels')
    save_pseudo_labels(pseudo_labels, confidence_mask, unlabeled_image_paths, 
                      pseudo_label_dir, model)
    
    print("Preparing updated dataset...")
    updated_images_dir = os.path.join(args.output_dir, 'dataset_updated/images')
    os.makedirs(updated_images_dir, exist_ok=True)
    
    for file in os.listdir(os.path.join(args.labeled_dir, 'images')):
        src = os.path.join(args.labeled_dir, 'images', file)
        dst = os.path.join(updated_images_dir, file)
        if os.path.isfile(src):
            shutil.copy(src, dst)
    
    for file in os.listdir(args.unlabeled_dir):
        src = os.path.join(args.unlabeled_dir, file)
        dst = os.path.join(updated_images_dir, file)
        if os.path.isfile(src):
            shutil.copy(src, dst)
    
    shutil.copytree(pseudo_label_dir, 
                    os.path.join(args.output_dir, 'dataset_updated/labels'), 
                    dirs_exist_ok=True)
    
    print("Starting final YOLO training...")
    model.train(
        data=args.dataset_updated_yaml,
        epochs=args.epochs,
        imgsz=512,
        batch=64,
        lr0=args.learning_rate,
        lrf=0.002,
        momentum=0.98,
        weight_decay=0.0001,
        optimizer="RAdam",
        cos_lr=True,
        warmup_epochs=30,
        iou=0.5,
        mosaic=0.0,
        dfl=2.0,
        augment=True,
        device="cuda"
    )
    
    print("Evaluating model...")
    metrics = model.val(data=args.test_yaml, device="cuda")
    precision = metrics.box.p.mean()
    recall = metrics.box.r.mean()
    map_50 = metrics.box.map50
    map_50_95 = metrics.box.map
    
    f1_score = (2 * precision * recall) / (precision + recall + 1e-6)
    final_score = 0.5 * f1_score + 0.5 * map_50_95
    
    print("\nFinal Metrics:")
    print(f"Mean Precision (mAP@50): {precision:.4f}")
    print(f"Mean Recall (mAP@50-95): {recall:.4f}")
    print(f"mAP@50: {map_50:.4f}")
    print(f"mAP@50-95: {map_50_95:.4f}")
    print(f"Mean F1-Score: {f1_score:.4f}")
    print(f"Final Score: {final_score:.4f}")
    
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Mean Precision (mAP@50): {precision:.4f}\n")
        f.write(f"Mean Recall (mAP@50-95): {recall:.4f}\n")
        f.write(f"mAP@50: {map_50:.4f}\n")
        f.write(f"mAP@50-95: {map_50_95:.4f}\n")
        f.write(f"Mean F1-Score: {f1_score:.4f}\n")
        f.write(f"Final Score: {final_score:.4f}\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise