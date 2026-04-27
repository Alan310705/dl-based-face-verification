import torch
import torch.nn.functional as F

class HardPairMiner:
    def __init__(self):
        pass

    def __call__(self, embeddings, labels):
        """
        Finds Hard Positive and Hard Negative pairs within a single batch.
        
        Args:
            embeddings: L2 normalized feature vectors (shape: batch_size, 1280)
            labels: Identity labels for each image in the batch (shape: batch_size)
            
        Returns:
            hard_pos_idx: Tensor of indices for the hardest positive pairs.
            hard_neg_idx: Tensor of indices for the hardest negative pairs.
        """
        # Ensure embeddings are L2 normalized to calculate Cosine Similarity via Dot Product
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Calculate the Cosine Similarity matrix (batch_size x batch_size)
        sim_matrix = torch.matmul(embeddings, embeddings.t())
        
        # Create masks for same-identity (Positive) and different-identity (Negative) pairs
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        # Remove the diagonal (preventing an image from being its own positive pair)
        device = embeddings.device
        mask_self = torch.eye(labels.size(0), dtype=torch.bool, device=device)
        labels_equal = labels_equal & ~mask_self
        labels_not_equal = ~labels_equal & ~mask_self
        
        # 1. Find HARD POSITIVES (Same identity but lowest similarity)
        sim_pos = sim_matrix.clone()
        # Assign a very large value (2.0) to NON-positive pairs so the .min() function ignores them
        sim_pos[~labels_equal] = 2.0 
        hard_pos_sim, hard_pos_idx = sim_pos.min(dim=1)
        
        # 2. Find HARD NEGATIVES (Different identity but highest similarity)
        sim_neg = sim_matrix.clone()
        # Assign a very small value (-2.0) to NON-negative pairs so the .max() function ignores them
        sim_neg[~labels_not_equal] = -2.0 
        hard_neg_sim, hard_neg_idx = sim_neg.max(dim=1)
        
        # Return the indices of the Hard Positive and Hard Negative for each image in the batch
        return hard_pos_idx, hard_neg_idx