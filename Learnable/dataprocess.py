import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from tqdm import tqdm
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class ImageNetDataset:
    """
    ImageNet dataset class with calibration and validation split functionality.
    Includes caching mechanism for faster data loading.
    """
    def __init__(self, args: Any):
        """
        Initialize ImageNet dataset with given arguments.
        
        Args:
            args: Arguments containing data_root, batch_size, num_workers, etc.
        """
        self.args = args
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def create_calibration_labels(
        self, 
        dataset: Dataset, 
        base_model: nn.Module
    ) -> torch.Tensor:
        """
        Create binary labels for calibration set based on model predictions.
        
        Args:
            dataset: Dataset to create calibration labels for
            base_model: Model to use for predictions
            
        Returns:
            Binary labels tensor (+1 for correct predictions, -1 for incorrect)
        """
        device = self.args.device
        base_model = base_model.to(device)
        base_model.eval()
        
        calib_batch_size = min(512, self.args.batch_size)

        all_labels = []
        loader = DataLoader(
            dataset, 
            batch_size=calib_batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        logger.info("Creating calibration labels...")

        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc="Creating calibration labels"):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs = base_model(inputs)
                predictions = outputs.argmax(dim=1)
                # Create binary labels: +1 if correct, -1 if incorrect
                binary_labels = 2 * (predictions == targets).float() - 1
                all_labels.append(binary_labels.cpu())
                
                del outputs, predictions
                torch.cuda.empty_cache()

        return torch.cat(all_labels)

    def split_validation_set(self, dataset: Dataset) -> Tuple[list, list]:
        """
        Split validation set into calibration and test sets.
        Ensures balanced splitting across classes.
        
        Args:
            dataset: Dataset to split
            
        Returns:
            Tuple of (calibration indices, validation indices)
        """
        total_size = len(dataset)
        indices = list(range(total_size))
        
        # Group indices by class
        class_indices = {}
        for idx in tqdm(indices, desc="Grouping by class"):
            label = dataset[idx][1]
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
            
        # Sample calibration indices from each class
        cal_indices = []
        val_indices = []
        
        for class_idx in tqdm(class_indices, desc="Splitting datasets"):
            class_size = len(class_indices[class_idx])
            cal_size = int(class_size * self.args.cal_size)
            
            # Randomly sample indices for calibration
            np.random.shuffle(class_indices[class_idx])
            cal_indices.extend(class_indices[class_idx][:cal_size])
            val_indices.extend(class_indices[class_idx][cal_size:])
            
        logger.info(f"Split sizes - Calibration: {len(cal_indices)}, "
                   f"Validation: {len(val_indices)}")
        return cal_indices, val_indices

    def save_calibration_data(
        self, 
        save_dir: str, 
        cal_indices: list, 
        val_indices: list, 
        binary_labels: torch.Tensor
    ) -> None:
        """
        Save calibration dataset information to files.
        
        Args:
            save_dir: Directory to save files
            cal_indices: Calibration set indices
            val_indices: Validation set indices
            binary_labels: Binary labels tensor
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save indices and binary labels
        np.save(save_path / 'cal_indices.npy', np.array(cal_indices))
        np.save(save_path / 'val_indices.npy', np.array(val_indices))
        torch.save(binary_labels, save_path / 'binary_labels.pt')
        
        # Save metadata
        metadata = {
            'cal_size': self.args.cal_size,
            'model_name': self.args.model_name,
            'num_cal_samples': len(cal_indices),
            'num_val_samples': len(val_indices)
        }
        torch.save(metadata, save_path / 'metadata.pt', _use_new_zipfile_serialization=True)
        
        logger.info(f"Saved calibration data to {save_path}")
        logger.info(f"Metadata: {metadata}")

    def load_calibration_data(
        self, 
        load_dir: str
    ) -> Optional[Tuple[list, list, torch.Tensor]]:
        """
        Load calibration dataset information from files if available.
        
        Args:
            load_dir: Directory to load files from
            
        Returns:
            Tuple of (cal_indices, val_indices, binary_labels) if successful,
            None otherwise
        """
        load_path = Path(load_dir)
        
        # Check if all required files exist
        required_files = [
            'cal_indices.npy',
            'val_indices.npy',
            'binary_labels.pt',
            'metadata.pt'
        ]
        
        if not all((load_path / f).exists() for f in required_files):
            logger.warning(f"Missing required files in {load_path}")
            return None
            
        # Load metadata and verify compatibility
        try:
            metadata = torch.load(load_path / 'metadata.pt', weights_only=True, 
                                map_location='cpu')
        except RuntimeError:
            metadata = torch.load(load_path / 'metadata.pt', map_location='cpu')

        if (metadata['cal_size'] != self.args.cal_size or 
            metadata['model_name'] != self.args.model_name):
            logger.warning("Cached calibration data parameters don't match current settings")
            logger.warning(f"Cached: {metadata}")
            logger.warning(f"Current: cal_size={self.args.cal_size}, "
                         f"model_name={self.args.model_name}")
            return None
            
        # Load the data
        cal_indices = np.load(load_path / 'cal_indices.npy').tolist()
        val_indices = np.load(load_path / 'val_indices.npy').tolist()

        try:
            binary_labels = torch.load(load_path / 'binary_labels.pt', weights_only=True,
                                    map_location='cpu')
        except RuntimeError:
            binary_labels = torch.load(load_path / 'binary_labels.pt', map_location='cpu')
        
        logger.info(f"Loaded calibration data from {load_path}")
        logger.info(f"Loaded data sizes - Calibration: {len(cal_indices)}, "
                   f"Validation: {len(val_indices)}")
        return cal_indices, val_indices, binary_labels

    def get_dataloaders(self, base_model: nn.Module) -> Dict[str, DataLoader]:
        """
        Create and return calibration and validation dataloaders.
        Will try to load from cache if available.
        
        Args:
            base_model: Model to use for creating calibration labels
            
        Returns:
            Dictionary containing 'calibration' and 'validation' dataloaders
        """
        # Load ImageNet validation set
        logger.info(f"Loading ImageNet from {self.args.data_root}")
        val_dataset = datasets.ImageNet(
            root=self.args.data_root,
            split='val',
            transform=self.transform
        )
        
        # Try to load cached calibration data if cache_dir is specified
        cached_data = None
        if hasattr(self.args, 'cache_dir') and self.args.cache_dir is not None:
            logger.info(f"Attempting to load cached data from {self.args.cache_dir}")
            cached_data = self.load_calibration_data(self.args.cache_dir)
        
        if cached_data is not None:
            logger.info("Using cached calibration data")
            cal_indices, val_indices, binary_labels = cached_data
        else:
            logger.info("Creating new calibration data")
            cal_indices, val_indices = self.split_validation_set(val_dataset)
            calibration_subset = Subset(val_dataset, cal_indices)

            # 
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                binary_labels = self.create_calibration_labels(calibration_subset, base_model)
            
            if torch.distributed.is_initialized():
                if torch.distributed.get_rank() == 0:
                    torch.distributed.broadcast_object_list([binary_labels], src=0)
                else:
                    binary_labels_list = [None]
                    torch.distributed.broadcast_object_list(binary_labels_list, src=0)
                    binary_labels = binary_labels_list[0]

            # Save the calibration data if cache_dir is specified
            if hasattr(self.args, 'cache_dir') and self.args.cache_dir is not None:
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    self.save_calibration_data(
                        self.args.cache_dir,
                        cal_indices,
                        val_indices,
                        binary_labels
                    )

        # Create datasets
        calibration_dataset = CalibrationDataset(
            Subset(val_dataset, cal_indices),
            binary_labels
        )
        validation_dataset = Subset(val_dataset, val_indices)
        
        # Create dataloaders with DistributedSampler if using distributed training
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            calibration_dataset
        ) if self.args.local_rank != -1 else None
        
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            validation_dataset, shuffle=False
        ) if self.args.local_rank != -1 else None
        
        dataloaders = {
            'calibration': DataLoader(
                calibration_dataset,
                batch_size=self.args.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=self.args.num_workers,
                pin_memory=True,
                drop_last=True  # Important for DDP
            ),
            'validation': DataLoader(
                validation_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=self.args.num_workers,
                pin_memory=True
            )
        }
        
        logger.info(f"Created dataloaders - "
                   f"Calibration size: {len(calibration_dataset)}, "
                   f"Validation size: {len(validation_dataset)}")
        return dataloaders

class CalibrationDataset(Dataset):
    """Dataset wrapper that includes binary labels for calibration."""
    
    def __init__(self, dataset: Dataset, binary_labels: torch.Tensor):
        """
        Initialize CalibrationDataset.
        
        Args:
            dataset: Base dataset to wrap
            binary_labels: Binary labels tensor (+1 for correct predictions, -1 for incorrect)
        """
        self.dataset = dataset
        self.binary_labels = binary_labels
        
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a dataset item with its binary label.
        
        Args:
            index: Index of the item to get
            
        Returns:
            Tuple of (input, target, binary_label)
        """
        inputs, targets = self.dataset[index]
        return inputs, targets, self.binary_labels[index]
        
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.dataset)