import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        # Shared backbone for both inputs
        self.backbone = backbone

    def forward(self, img_a, img_b):
        # Extract embeddings using the shared weights
        emb_a = self.backbone(img_a)
        emb_b = self.backbone(img_b)

        # Calculate Cosine Similarity
        cos_sim = F.cosine_similarity(emb_a, emb_b, dim=1)

        return emb_a, emb_b, cos_sim