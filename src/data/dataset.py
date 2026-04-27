import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------
# 1. TRAINING DATASET (For ArcFace & Hard Pair Mining)
# ---------------------------------------------------------
class LFWTrainingDataset(Dataset):
    def __init__(self, processed_data_dir, img_size='m', transform=None, min_images=4):
        """
        Loads individual images for training. Dynamically filters out identities 
        that do not have enough images to form meaningful pairs.
        """
        self.img_dir = os.path.join(processed_data_dir, img_size)
        
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.samples = []
        self.labels = []
        self.label_to_indices = {} 
        
        # Build the dataset, filtering out identities with < min_images
        identities = sorted(os.listdir(self.img_dir))
        current_label_id = 0
        
        for identity in identities:
            identity_dir = os.path.join(self.img_dir, identity)
            if not os.path.isdir(identity_dir):
                continue
                
            images = os.listdir(identity_dir)
            
            # DYNAMIC FILTERING: Skip identities with too few images
            if len(images) < min_images:
                continue 
                
            self.label_to_indices[current_label_id] = []
            
            for img_name in images:
                self.samples.append(os.path.join(identity_dir, img_name))
                self.labels.append(current_label_id)
                self.label_to_indices[current_label_id].append(len(self.samples) - 1)
                
            current_label_id += 1
            
        print(f"[INFO] Training Dataset: Kept {current_label_id} identities, {len(self.samples)} total images.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(label, dtype=torch.long)

# ---------------------------------------------------------
# 2. PK BATCH SAMPLER (The Engine for OHPM)
# ---------------------------------------------------------
class PKBatchSampler(Sampler):
    def __init__(self, dataset, p=16, k=4):
        """
        Ensures every batch has P identities, and K images per identity.
        Batch Size = P * K (e.g., 16 * 4 = 64).
        """
        self.dataset = dataset
        self.p = p
        self.k = k
        self.batch_size = p * k
        self.label_to_indices = dataset.label_to_indices
        self.available_labels = list(self.label_to_indices.keys())
        
        # Calculate how many batches make up one full epoch
        self.num_batches = len(self.dataset) // self.batch_size

    def __iter__(self):
        for _ in range(self.num_batches):
            batch_indices = []
            
            # 1. Randomly select P identities for this batch
            selected_classes = random.sample(self.available_labels, self.p)
            
            # 2. For each identity, select K images
            for cls in selected_classes:
                indices = self.label_to_indices[cls]
                
                # Randomly sample K indices. If the person has exactly K images, 
                # replace=False uses them all. If they have more, it picks K randomly.
                selected_indices = np.random.choice(indices, self.k, replace=False)
                batch_indices.extend(selected_indices.tolist())
                
            yield batch_indices

    def __len__(self):
        return self.num_batches

# ---------------------------------------------------------
# 3. EVALUATION DATASET
# ---------------------------------------------------------
class LFWPairsDataset(Dataset):
    def __init__(self, pairs_csv_path, processed_data_dir, img_size='m', transform=None):
        """
        Loads the exact 6,000 pairs from LFW for ROC/EER evaluation.
        """
        self.pairs_csv_path = pairs_csv_path
        self.img_dir = os.path.join(processed_data_dir, img_size)
        self.pairs = self._parse_pairs_csv()
        
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _parse_pairs_csv(self):
        pairs = []
        with open(self.pairs_csv_path, 'r') as f:
            lines = f.readlines()
            
        # Skip header line
        for line in lines[1:]:
            parts = line.strip().replace(',', ' ').split() 
            
            if not parts:
                continue
                
            if len(parts) == 3:
                # Match pair: name, img1_id, img2_id
                name, id1, id2 = parts
                img1_path = os.path.join(self.img_dir, name, f"{name}_{int(id1):04d}.jpg")
                img2_path = os.path.join(self.img_dir, name, f"{name}_{int(id2):04d}.jpg")
                pairs.append((img1_path, img2_path, 1)) # 1: Same person
                
            elif len(parts) == 4:
                # Mismatch pair: name1, id1, name2, id2
                name1, id1, name2, id2 = parts
                img1_path = os.path.join(self.img_dir, name1, f"{name1}_{int(id1):04d}.jpg")
                img2_path = os.path.join(self.img_dir, name2, f"{name2}_{int(id2):04d}.jpg")
                pairs.append((img1_path, img2_path, 0)) # 0: Different people
                
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img1, img2, torch.tensor(label, dtype=torch.float32)