import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import gc

# Import local modules
from src.data.dataset import LFWPairsDataset
from src.models.backbone import EfficientNetV2Backbone
from src.models.siamese import SiameseNetwork
from src.models.losses import ContrastiveLoss

def train_baselines():
    # 1. Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Starting training on device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU Name: {torch.cuda.get_device_name(0)}")

    # Create checkpoints directory
    os.makedirs('outputs/checkpoints', exist_ok=True)

    # 2. Configuration (Variant, Batch Size)
    configs = [
        ('s', 16),
        ('m', 8), 
        ('l', 4)  
    ]
    
    num_epochs = 10
    learning_rate = 1e-4
    
    # Data paths
    pairs_file = 'src/data/data/raw/pairs.csv' 
    processed_dir = 'src/data/data/processed'

    for variant, batch_size in configs:
        print(f"\n{'='*60}")
        print(f"[INFO] TRAINING START: EfficientNetV2-{variant.upper()} | Batch Size: {batch_size}")
        print(f"{'='*60}")

        # 3. Initialize Dataset and DataLoader
        try:
            train_dataset = LFWPairsDataset(
                pairs_csv_path=pairs_file,
                processed_data_dir=processed_dir,
                img_size=variant
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            print(f"[INFO] Successfully loaded {len(train_dataset)} pairs from LFW.")
        except Exception as e:
            print(f"[ERROR] Failed to load data for variant {variant}: {e}")
            continue

        # 4. Initialize Model, Loss, Optimizer
        print("[INFO] Loading weights and moving model to GPU...")
        backbone = EfficientNetV2Backbone(variant=variant)
        model = SiameseNetwork(backbone).to(device)
        
        criterion = ContrastiveLoss(margin=1.0)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 5. Training Loop
        model.train()
        for epoch in range(num_epochs):
            start_time = time.time()
            total_loss = 0.0
            
            for batch_idx, (img1, img2, labels) in enumerate(train_loader):
                # Move data to GPU
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                emb_a, emb_b, cos_sim = model(img1, img2)

                # Compute loss
                loss = criterion(emb_a, emb_b, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Log epoch results
            avg_loss = total_loss / len(train_loader)
            epoch_time = time.time() - start_time
            print(f"[INFO] Epoch [{epoch+1:02d}/{num_epochs}] | Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")

        # 6. Save Checkpoint
        save_path = f'outputs/checkpoints/baseline_{variant}.pt'
        torch.save(model.state_dict(), save_path)
        print(f"[INFO] Weights successfully saved at: {save_path}")

        # 7. Clear GPU memory before next variant
        del model, backbone, optimizer, train_loader, train_dataset
        gc.collect() 
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 

if __name__ == "__main__":
    train_baselines()