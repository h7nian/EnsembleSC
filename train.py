import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
import logging

class SelectiveTrainer:
    def __init__(self,
                 model: nn.Module,
                 learning_rate: float = 0.001,
                 device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Optimizer only for selector weights
        self.optimizer = optim.Adam(
            [model.selector_weights],
            lr=learning_rate
        )
        
        self.logger = logging.getLogger(__name__) 
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_scores = []
        all_labels = []
        
        for x, y_s in tqdm(train_loader, desc='Training'):
            x, y_s = x.to(self.device), y_s.to(self.device)
            
            # Get logits and combined scores directly from model forward pass
            _, combined_scores = self.model(x)
            
            # Calculate hinge loss
            loss = torch.nn.functional.hinge_embedding_loss(
                combined_scores,
                y_s,
                margin=1.0,
                reduction='mean'
            )
            
            # Update weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            all_scores.append(combined_scores.detach().cpu())
            all_labels.append(y_s.cpu())
            
        # Calculate RC curve for training
        all_scores = torch.cat(all_scores)
        all_labels = torch.cat(all_labels)
        rc_metrics = self.calculate_rc_curve(train_loader)  # Pass the whole loader
        
        # Add average loss to metrics
        rc_metrics['loss'] = total_loss / len(train_loader)
        
        return rc_metrics
    
    def calculate_rc_curve(self, data_loader: DataLoader, num_thresholds: int = 10) -> Dict[str, np.ndarray]:
        """Calculate risk-coverage curve using Selective Risk formula"""
        self.model.eval()
        
        # First pass: collect selector scores
        all_scores = []
        with torch.no_grad():
            for x, y in tqdm(data_loader, desc="Collecting scores"): 
                x = x.to(self.device)
                _, selector_scores = self.model(x)
                all_scores.append(selector_scores.cpu())
        all_scores = torch.cat(all_scores)
        
        min_score = all_scores.min().item()
        max_score = all_scores.max().item()
        thresholds = np.linspace(min_score, max_score, num_thresholds)
        
        coverages = []  
        risks = []

        for threshold in tqdm(thresholds, desc="Computing risks"):
            error_sum = 0
            total_samples = 0
            
            # Second pass: process samples in batches
            with torch.amp.autocast('cuda'):  
                with torch.no_grad():
                    for x, y in data_loader:
                        batch_size = len(x)
                        x = x.to(self.device)
                        y = y.to(self.device)
                        
                        batch_scores = all_scores[total_samples:total_samples + batch_size].to(self.device)
                        gs_gamma = (batch_scores >= threshold).float()
                        
                        base_logits = self.model.base_model(x)
                        base_preds = base_logits.argmax(dim=1)
                        
                        errors = (base_preds != y).float()
                        error_sum += (errors * gs_gamma).sum().item()
                        total_samples += batch_size
                        
                        torch.cuda.empty_cache()
            
            gs_gamma_all = (all_scores >= threshold).float()
            coverage = gs_gamma_all.mean().item()
            
            if coverage > 0:
                risk = error_sum / (coverage * total_samples)
            else:
                risk = 1.0
                
            coverages.append(coverage)
            risks.append(risk)

            self.logger.info(f"Threshold: {threshold:.4f}, Coverage: {coverage:.4f}, Risk: {risk:.4f}")
            
            if len(coverages) % 10 == 0:
                torch.cuda.empty_cache()
        
        return {
            'thresholds': np.array(thresholds),
            'coverages': np.array(coverages),
            'risks': np.array(risks)
        }
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, np.ndarray]:
        """Evaluate model and compute RC curve"""
        self.model.eval()
        return self.calculate_rc_curve(val_loader)
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              n_epochs: int) -> Dict[str, List[Dict[str, np.ndarray]]]:
        """Complete training process with validation"""
        train_metrics = []
        val_metrics = []
        
        for epoch in range(n_epochs):
            self.logger.info(f"Epoch {epoch+1}/{n_epochs}")
            
            # Training
            train_metric = self.train_epoch(train_loader)
            train_metrics.append(train_metric)
            
            # Validation
            val_metric = self.evaluate(val_loader)
            val_metrics.append(val_metric)
            
            # Log metrics at specific coverage points
            coverage_points = [0.8, 0.9, 0.95]
            for coverage in coverage_points:
                idx = np.argmin(np.abs(np.array(val_metric['coverages']) - coverage))
                self.logger.info(
                    f"Val metrics at {coverage:.2f} coverage: "
                    f"risk={val_metric['risks'][idx]:.4f}, "
                    f"threshold={val_metric['thresholds'][idx]:.4f}"
                )
            
        return {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }