import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np

from utils import calculate_rc_curve, plot_combined_rc_curves, evaluate_at_coverage

logger = logging.getLogger(__name__)

class SelectiveTrainer:
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
        device: str,
        output_dir: Path,
        base_metrics: Dict,
        val_interval: int = 1,
        model_type: str = "EnsembleSC",
        args: Any = None
    ):
        
        self.args = args
        self.device = device
        self.output_dir = output_dir
        self.base_metrics = base_metrics
        self.val_interval = val_interval
        self.model_type = model_type

        model = model.to(device)

        if args and args.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True 
            )
        elif torch.cuda.device_count() > 1:
            if args and args.local_rank in [-1, 0]:
                logger.info(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(model)
        else:
            self.model = model

        model_to_optimize = self.model.module if isinstance(
            self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)
        ) else self.model

        if model_type == "EnsembleSC":
            optimizer_params = list(model_to_optimize.score_network.parameters())
        elif model_type == "EnsembleSC-Learnable": 
            optimizer_params = [
                {'params': model_to_optimize.score_network.parameters()},
                {'params': model_to_optimize.selector_thresholds}
            ]
        
        self.optimizer = torch.optim.Adam(optimizer_params, lr=learning_rate)
    
    def evaluate(self, val_loader: DataLoader, threshold: float = None) -> Dict[str, Any]:
        """Evaluate the model."""
        self.model.eval()
        all_scores = []
        all_correct = []
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for inputs, targets in tqdm(val_loader, desc="Evaluating",
                                      disable=not (not torch.distributed.is_initialized() 
                                                 or torch.distributed.get_rank() == 0)):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                logits, scores = self.model(inputs)
                predictions = logits.argmax(dim=1)
                correct = predictions == targets
                
                all_scores.extend(scores.cpu().numpy())
                all_correct.extend(correct.cpu().numpy())
                
                del logits, scores, predictions, correct
                torch.cuda.empty_cache()
        
        coverages, risks, thresholds = calculate_rc_curve(all_scores, all_correct)
        
        metrics = {
            'scores': np.array(all_scores),
            'correct': np.array(all_correct),
            'coverages': coverages,
            'risks': risks,
            'thresholds': thresholds
        }
        
        return metrics
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
            """Train for one epoch."""
            self.model.train()
            total_loss = 0
            all_scores = []
            all_correct = []
            
            for inputs, targets, binary_labels in tqdm(train_loader, desc="Training",
                                                    disable=not (not torch.distributed.is_initialized() 
                                                                or torch.distributed.get_rank() == 0)):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                binary_labels = binary_labels.to(self.device)
                
                logits, selection_scores = self.model(inputs)
                
                # Compute hinge loss
                hinge_loss = torch.mean(torch.clamp(1 - binary_labels * selection_scores, min=0))
                
                self.optimizer.zero_grad()
                hinge_loss.backward()
                self.optimizer.step()
                
                total_loss += hinge_loss.item()
                predictions = logits.argmax(dim=1)
                correct = predictions == targets
                
                all_scores.extend(selection_scores.detach().cpu().numpy())
                all_correct.extend(correct.cpu().numpy())

                del logits, selection_scores, predictions, correct
                torch.cuda.empty_cache()
            
            # Calculate metrics using risk-coverage curve
            coverages, risks, _ = calculate_rc_curve(all_scores, all_correct)
            
            metrics = {
                'loss': total_loss / len(train_loader),
                'risk': np.mean(risks),
                'coverage': np.mean(coverages)
            }
            
            return metrics

    def train(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            n_epochs: int,
            target_coverage: float = 0.8
        ) -> Dict[str, Any]:
            """Train the model for multiple epochs."""
            best_metrics = None
            train_metrics_history = []
            val_metrics_history = []
            best_risk = float('inf')
            
            for epoch in range(n_epochs):
                logger.info(f"\nEpoch {epoch+1}/{n_epochs}")
                
                if hasattr(train_loader, 'sampler') and isinstance(
                    train_loader.sampler, torch.utils.data.distributed.DistributedSampler
                ):
                    train_loader.sampler.set_epoch(epoch)
                
                # Train
                train_metrics = self.train_epoch(train_loader)
                train_metrics_history.append(train_metrics)
                if self.args.local_rank in [-1, 0]:
                    logger.info(f"\nEpoch {epoch+1}/{n_epochs}")
                    logger.info(f"Train - Loss: {train_metrics['loss']:.4f}")
                    
                    # Print selector weights
                    model_to_print = (
                        self.model.module if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel))
                        else self.model
                    )
                    selector_importance = model_to_print.get_selector_importance()
                    logger.info("\nSelector Importance:")
                    for name, importance in selector_importance:
                        logger.info(f"{name:10s}: {importance:.4f}")
                        
                # Evaluate periodically
                if (epoch + 1) % self.val_interval == 0:
                    if torch.distributed.is_initialized():
                        torch.distributed.barrier()
                    
                    val_metrics = self.evaluate(val_loader)
                    
                    if self.args.local_rank in [-1, 0]:
                        coverages, risks, thresholds = calculate_rc_curve(
                            scores=val_metrics['scores'],
                            correct=val_metrics['correct']
                        )
                        
                        ensemble_metrics = evaluate_at_coverage(coverages, risks, thresholds, target_coverage)
                        logger.info(f"Ensemble - Risk: {ensemble_metrics['risk']:.4f}, "
                                f"Coverage: {ensemble_metrics['coverage']:.4f}")
                        
                        val_metrics.update({
                            'threshold': ensemble_metrics['threshold'],
                            'selected_risk': ensemble_metrics['risk'],
                            'achieved_coverage': ensemble_metrics['coverage']
                        })
                        
                        for name, metrics in self.base_metrics.items():
                            target_metrics = evaluate_at_coverage(
                                metrics['coverages'], 
                                metrics['risks'], 
                                metrics['thresholds'], 
                                target_coverage
                            )
                            logger.info(f"{name:10s} - Risk: {target_metrics['risk']:.4f}, "
                                    f"Coverage: {target_metrics['coverage']:.4f}")
                        
                        val_metrics_history.append(val_metrics)

                        if ensemble_metrics['risk'] < best_risk:
                            best_risk = ensemble_metrics['risk']
                            best_metrics = val_metrics
                            
                            model_to_save = (
                                self.model.module if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel))
                                else self.model
                            )
                            torch.save({
                                'model_state_dict': model_to_save.state_dict(),
                                'metrics': best_metrics,
                                'threshold': ensemble_metrics['threshold']
                            }, self.output_dir / f'{self.model_type}_best_model.pt')
            
            return {
                'train_history': train_metrics_history,
                'val_history': val_metrics_history,
                'best_metrics': best_metrics
            }