import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from typing import Tuple, Dict, List
import random
from tqdm import tqdm
class ImageNetDataset:
    def __init__(self, args):
        self.args = args
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.val_dataset = datasets.ImageNet(
            root=args.data_root,
            split='val',
            transform=self.transform
        )
        
    def create_splits(self, base_model: nn.Module, cal_size: float = 0.4) -> Tuple[Dataset, Dataset]:
        """Split validation set into new validation and calibration sets"""
        total_size = len(self.val_dataset)
        indices = list(range(total_size))
        random.shuffle(indices)
        
        split = int(total_size * cal_size)
        cal_indices = indices[:split]
        val_indices = indices[split:]
        
        # Create calibration dataset with binary labels
        cal_dataset = CalibrationDataset(
            dataset=self.val_dataset,
            indices=cal_indices,
            base_model=base_model,
            device=self.args.device
        )
        
        # Create new validation dataset
        val_dataset = Subset(self.val_dataset, val_indices)
        
        return cal_dataset, val_dataset
        
    def get_dataloaders(self, base_model: nn.Module) -> Dict[str, DataLoader]:
        """Create dataloaders for calibration and validation sets"""

        cal_dataset, val_dataset = self.create_splits(
            base_model=base_model,
            cal_size=self.args.cal_size
        )
        
        cal_loader = DataLoader(
            cal_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers
        )
        
        return {
            'calibration': cal_loader,
            'validation': val_loader
        }

class CalibrationDataset(Dataset):
    """Dataset wrapper that converts model predictions to binary labels"""
    def __init__(self, dataset: Dataset, indices: List[int], base_model: nn.Module, device: str):
        self.dataset = dataset
        self.indices = indices
        self.device = device
        
        # Prepare shifted data using base model predictions
        temp_loader = DataLoader(
            Subset(dataset, indices),
            batch_size=1024,
            shuffle=False
        )
        self.features, self.binary_labels = prepare_shifted_data(base_model, temp_loader, device)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, _ = self.dataset[self.indices[idx]]
        binary_label = self.binary_labels[idx]
        return x, binary_label
        
    def __len__(self) -> int:
        return len(self.indices)
        
def prepare_shifted_data(base_model: nn.Module,
                        data_loader: DataLoader,
                        device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare dataset with binary labels indicating if base model prediction was correct"""
    all_features = []
    binary_labels = []
    
    base_model = base_model.to(device)
    base_model.eval()
    
    with torch.no_grad():
        for x, y in tqdm(data_loader,desc = "Calibration Dataloader"):
            x = x.to(device)
            y = y.to(device)
            
            # Get model predictions
            logits = base_model(x)
            preds = logits.argmax(dim=1)
            
            # Create binary labels (1 if correct, -1 if wrong)
            binary = 2 * (preds == y).float() - 1
            
            all_features.append(logits.cpu())
            binary_labels.append(binary.cpu())
    
    return torch.cat(all_features), torch.cat(binary_labels)