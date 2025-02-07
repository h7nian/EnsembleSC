import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any
import logging
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)

class SelectiveTrainer:
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
        device: str,
        val_interval: int = 1  # Add validation interval parameter
    ):
        self.model = model.to(device)
        self.device = device
        self.val_interval = val_interval
        self.optimizer = torch.optim.Adam([
            {'params': self.model.selector_weights},
            {'params': self.model.selector_thresholds},
            {'params': self.model.ensemble_threshold}
        ], lr=learning_rate)
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        selected = 0
        total = 0
        
        for inputs, targets, binary_labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            binary_labels = binary_labels.to(self.device)
            
            # Forward pass
            logits, selection_scores = self.model(inputs)
            
            # Compute hinge loss
            hinge_loss = torch.mean(torch.clamp(1 - binary_labels * selection_scores, min=0))
            
            # Backward pass
            self.optimizer.zero_grad()
            hinge_loss.backward()
            self.optimizer.step()
            
            # Compute metrics
            total_loss += hinge_loss.item()
            predictions = logits.argmax(dim=1)
            selected_mask = selection_scores > 0
            
            correct += ((predictions == targets) & selected_mask).sum().item()
            selected += selected_mask.sum().item()
            total += len(inputs)
            
        # Compute epoch metrics
        metrics = {
            'loss': total_loss / len(train_loader),
            'accuracy': correct / selected if selected > 0 else 0,
            'coverage': selected / total
        }
        
        return metrics
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, Any]:
        """Evaluate the model including individual selector performances."""
        self.model.eval()
        correct = 0
        selected = 0
        total = 0
        
        # Store scores for each selector and ensemble
        selector_names = ['sr-max', 'sr-doctor', 'sr-entropy', 'rl-geom', 'rl-confm', 'ensemble']
        all_scores = {name: [] for name in selector_names}
        all_correct = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Evaluating"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Get base model predictions and individual selector scores
                logits = self.model.base_model(inputs)
                predictions = logits.argmax(dim=1)
                correct_mask = predictions == targets
                
                # Get individual selector scores
                for idx, selector in enumerate(self.model.selectors):
                    scores = selector(logits)
                    # normalized_scores = torch.tanh(scores - self.model.selector_thresholds[idx])
                    normalized_scores = scores - self.model.selector_thresholds[idx]
                    all_scores[selector_names[idx]].extend(normalized_scores.cpu().numpy())
                
                # Get ensemble scores
                _, _, selection_scores = self.model.predict_with_selection(inputs)
                all_scores['ensemble'].extend(selection_scores.cpu().numpy())
                
                # Store correctness
                all_correct.extend(correct_mask.cpu().numpy())
                
                # Compute metrics for ensemble selection
                selected_mask = selection_scores > 0
                correct += (correct_mask & selected_mask).sum().item()
                selected += selected_mask.sum().item()
                total += len(inputs)
        
        # Create metrics dictionary
        metrics = {
            'accuracy': correct / selected if selected > 0 else 0,
            'coverage': selected / total,
            'correct': all_correct
        }
        
        # Add scores for each selector
        for name in selector_names:
            metrics[f'{name}_scores'] = all_scores[name]
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int
    ) -> Dict[str, Any]:
        """Train the model for multiple epochs."""
        best_metrics = None
        train_metrics_history = []
        val_metrics_history = []
        
        for epoch in range(n_epochs):
            logger.info(f"\nEpoch {epoch+1}/{n_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                       f"Accuracy: {train_metrics['accuracy']:.4f}, "
                       f"Coverage: {train_metrics['coverage']:.4f}")
            
            # Store metrics
            train_metrics_history.append(train_metrics)
            
            # Evaluate periodically
            if (epoch + 1) % self.val_interval == 0:
                val_metrics = self.evaluate(val_loader)
                logger.info(f"Val - Accuracy: {val_metrics['accuracy']:.4f}, "
                          f"Coverage: {val_metrics['coverage']:.4f}")
                val_metrics_history.append(val_metrics)
                
                # Update best metrics
                if (best_metrics is None or 
                    val_metrics['accuracy'] > best_metrics['accuracy']):
                    best_metrics = val_metrics
        
        return {
            'train_history': train_metrics_history,
            'val_history': val_metrics_history,
            'best_metrics': best_metrics
        }