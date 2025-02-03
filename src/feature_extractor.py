import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, image_paths):
        features = []
        for img_path in image_paths:
            img = self.transform(Image.open(img_path).convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                feat1 = self.model.model.model[0:5](img)
                feat2 = self.model.model.model[0:10](img)
                
                feat1_pooled = torch.cat([
                    torch.mean(feat1, dim=[2, 3]),
                    torch.max(feat1, dim=3)[0].max(dim=2)[0]
                ], dim=1)
                
                feat2_pooled = torch.cat([
                    torch.mean(feat2, dim=[2, 3]),
                    torch.max(feat2, dim=3)[0].max(dim=2)[0]
                ], dim=1)
                
                combined_feat = torch.cat([feat1_pooled, feat2_pooled], dim=1)
                features.append(combined_feat.view(-1).numpy())
        
        features = np.array(features)
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)
        return features