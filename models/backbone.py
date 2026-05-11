import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class EfficientNetV2Backbone(nn.Module):
    def __init__(self, variant='s', unfreeze_ratio=0.3):
        super().__init__()
        self.model_name = f'tf_efficientnetv2_{variant}'
        
        # Drop classifier head (num_classes=0) for feature extraction
        self.backbone = timm.create_model(self.model_name, pretrained=True, num_classes=0)
        
        # Setup layer freezing
        self._freeze_layers(unfreeze_ratio)

    def _freeze_layers(self, unfreeze_ratio):
        parameters = list(self.backbone.parameters())
        total_params = len(parameters)
        
        # Determine layer cutoff point
        freeze_cutoff = int(total_params * (1.0 - unfreeze_ratio))
        
        for i, param in enumerate(parameters):
            if i < freeze_cutoff:
                param.requires_grad = False  # Freeze
            else:
                param.requires_grad = True   # Unfreeze
                
        print(f"[INFO] {self.model_name}: Froze the first {freeze_cutoff}/{total_params} layers.")
        print(f"[INFO] {self.model_name}: Unfroze the last {total_params - freeze_cutoff} layers for fine-tuning.")

    def forward(self, x):
        features = self.backbone(x)
        # L2 Normalization
        embeddings = F.normalize(features, p=2, dim=1)
        return embeddings