import argparse
from pathlib import Path
import torch
import logging
import numpy as np
from dataprocess import ImageNetDataset
from models import create_selective_network, SRMaxSelector, SRDoctorSelector, SREntropySelector, RLGeoMSelector, RLConfMSelector
from train import SelectiveTrainer
from torch.utils.data import DataLoader
from utils import setup_logging, save_metrics, calculate_rc_curve, plot_combined_rc_curves
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import json
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Train Selective Classification')
    
    # Dataset related arguments
    parser.add_argument('--data-root', type=str, default='/users/0/zhan9381/EnsembleSC/Codes/ImageNet',
                      help='Path to ImageNet dataset')
    parser.add_argument('--cal-size', type=float, default=0.1,
                      help='Proportion of data for calibration')
    parser.add_argument('--batch-size', type=int, default=1024,
                      help='Batch size')
    parser.add_argument('--num-workers', type=int, default=2,
                      help='Number of data loading workers')
    
    # Model related arguments
    parser.add_argument('--model-name', type=str,
                      default='eva_giant_patch14_224.clip_ft_in1k',
                      help='Name of the pretrained model')
    
    # Training related arguments
    parser.add_argument('--target-coverage', type=float, default=0.8,
                      help='Target coverage rate')
    parser.add_argument('--n-epochs', type=int, default=10,
                      help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--val-interval', type=int, default=1,
                      help='Validation interval (epochs)')
    
    # Output related arguments
    parser.add_argument('--output-dir', type=str, default='outputs',
                      help='Directory to save outputs')
    parser.add_argument('--cache-dir', type=str, default='data',
                      help='Directory to save/load calibration data. If not specified, will not use cache.')
    parser.add_argument('--experiment-name', type=str, default='CombinedTraining',
                      help='Name of the experiment for organizing outputs')
    
    # distributed training arguments
    parser.add_argument('--local-rank', type=int, default=-1,
                      help='Local rank for distributed training')
    parser.add_argument('--world-size', type=int, default=-1,
                      help='Number of distributed processes')
    parser.add_argument('--dist-url', type=str, default='env://',
                      help='URL used to set up distributed training')
    
    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, args.experiment_name)
    
    if args.local_rank in [-1, 0]:
        if args.cache_dir is not None:
            args.cache_dir = os.path.join(args.cache_dir, args.experiment_name)
            os.makedirs(args.cache_dir, exist_ok=True)
        
    return args

def evaluate_base_selector(selector, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
    """Evaluate a single selector"""
    with torch.no_grad():
        # Get predictions and scores
        predictions = logits.argmax(dim=1)
        scores = selector(logits)
        correct = predictions == targets

        scores = scores.cpu()
        correct = correct.cpu()
        del predictions
        torch.cuda.empty_cache()
        
    return {
        'scores': scores.cpu().numpy(),
        'correct': correct.cpu().numpy()
    }

def evaluate_base_selectors(model, val_loader: DataLoader, device: str) -> Dict[str, Any]:
    """Evaluate individual base selectors with memory efficient processing"""
    logger = logging.getLogger(__name__)
    
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.info("Evaluating base selectors...")
    
    # Initialize base selectors
    base_selectors = {
        'SR-Max': SRMaxSelector(),
        'SR-Doctor': SRDoctorSelector(),
        'SR-Entropy': SREntropySelector(),
        'RL-GeoM': RLGeoMSelector(model),
        'RL-ConfM': RLConfMSelector()
    }

    model = model.to(device)
    model.eval()
    
    all_metrics = {name: {'scores': [], 'correct': []} for name in base_selectors.keys()}
    
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for inputs, targets in tqdm(val_loader, desc="Evaluating base selectors",
                                  disable=not (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)):
            torch.cuda.empty_cache()
            
            try:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                logits = model(inputs)
                
                for name, selector in base_selectors.items():
                    metrics = evaluate_base_selector(selector, logits, targets)
                    all_metrics[name]['scores'].extend(metrics['scores'])
                    all_metrics[name]['correct'].extend(metrics['correct'])
                
                del logits
                del inputs
                del targets
                torch.cuda.empty_cache()
                
            except torch.cuda.OutOfMemoryError as e:
                del logits, inputs, targets
                torch.cuda.empty_cache()
                logger.warning(f"Out of memory error occurred. Trying to recover...")
                continue
    
    if torch.distributed.is_initialized():
        for name in base_selectors.keys():
            gathered_scores = [None for _ in range(torch.distributed.get_world_size())]
            gathered_correct = [None for _ in range(torch.distributed.get_world_size())]

            local_scores = torch.tensor(all_metrics[name]['scores'], device=device)
            local_correct = torch.tensor(all_metrics[name]['correct'], device=device)
            
            torch.distributed.all_gather(gathered_scores, local_scores)
            torch.distributed.all_gather(gathered_correct, local_correct)
            
            if torch.distributed.get_rank() == 0:
                all_scores = np.concatenate([t.cpu().numpy() for t in gathered_scores])
                all_correct = np.concatenate([t.cpu().numpy() for t in gathered_correct])
                
                coverages, risks, thresholds = calculate_rc_curve(all_scores, all_correct)
                
                all_metrics[name] = {
                    'scores': all_scores,
                    'correct': all_correct,
                    'coverages': coverages,
                    'risks': risks,
                    'thresholds': thresholds
                }
    else:

        for name in base_selectors.keys():
            scores = np.array(all_metrics[name]['scores'])
            correct = np.array(all_metrics[name]['correct'])
            
            coverages, risks, thresholds = calculate_rc_curve(scores, correct)
            all_metrics[name] = {
                'scores': scores,
                'correct': correct,
                'coverages': coverages,
                'risks': risks,
                'thresholds': thresholds
            }
    
    return all_metrics
def train_and_evaluate_model(model_type: str, args: argparse.Namespace, 
                           dataset: ImageNetDataset, output_dir: Path, base_metrics: Dict) -> Dict[str, Any]:
    """Train and evaluate a specific model type"""
    logger = logging.getLogger(__name__)
    
    # Create model
    logger.info(f"Creating {model_type} model...")
    model = create_selective_network(args.model_name, model_type)
    
    # Create model-specific output directory
    model_dir = output_dir / model_type
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create trainer
    trainer = SelectiveTrainer(
        model=model,
        learning_rate=args.lr,
        device=args.device,
        output_dir=model_dir,
        base_metrics = base_metrics,
        val_interval=args.val_interval,
        model_type=model_type,
        args=args
    )
    
    # Get dataloaders
    dataloaders = dataset.get_dataloaders(base_model=model.base_model)
    
    # Train model
    logger.info(f"Training {model_type}...")
    train_metrics = trainer.train(
        train_loader=dataloaders['calibration'],
        val_loader=dataloaders['validation'],
        n_epochs=args.n_epochs,
        target_coverage=args.target_coverage
    )
    
    # Final evaluation
    logger.info(f"Final evaluation of {model_type}...")
    final_metrics = trainer.evaluate(dataloaders['validation'])
    
    # Calculate RC curve
    coverages, risks, thresholds = calculate_rc_curve(
        final_metrics['scores'],
        final_metrics['correct']
    )
    
    final_metrics.update({
        'coverages': coverages,
        'risks': risks,
        'thresholds': thresholds
    })
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'metrics': final_metrics,
        'args': vars(args)
    }, model_dir / f'{model_type}_model.pt')
    
    return final_metrics

def setup_distributed(args):
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
    else:
        args.world_size = 1

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup distributed training
    setup_distributed(args)
    
    # Set device
    args.device = f'cuda:{args.local_rank}' if args.local_rank != -1 else 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Create output directory and setup logging only on master process
    if args.local_rank in [-1, 0]:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(str(output_dir))
        logger = logging.getLogger(__name__)
        logger.info(f"Arguments: {args}")
        logger.info(f"Device: {args.device}")
    else:
        logger = logging.getLogger(__name__)
        output_dir = Path(args.output_dir)
    
    # Wait for master process to create directories
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    # Initialize base model for base selectors evaluation
    if args.local_rank in [-1, 0]:
        logger.info("Initializing base model...")
    base_model = create_selective_network(args.model_name, "EnsembleSC").base_model
    base_model = base_model.to(args.device)
    
    # Create dataset
    if args.local_rank in [-1, 0]:
        logger.info("Creating dataset...")
    dataset = ImageNetDataset(args)
    dataloaders = dataset.get_dataloaders(base_model=base_model)
    
    # Ensure all processes have initialized their dataloaders
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    # Store all results
    all_metrics = {}
    
    # 1. Evaluate base selectors
    if args.local_rank in [-1, 0]:
        logger.info("Evaluating base selectors...")
    base_metrics = evaluate_base_selectors(base_model, dataloaders['validation'], args.device)
    
    # Ensure base selector evaluation is complete on all processes
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    all_metrics.update(base_metrics)
    
    # 2. Train and evaluate ensemble models
    model_types = ["EnsembleSC-Learnable", "EnsembleSC"]
    for model_type in model_types:
        try:
            if args.local_rank in [-1, 0]:
                logger.info(f"Processing {model_type}...")
            metrics = train_and_evaluate_model(model_type, args, dataset, output_dir, base_metrics)
            all_metrics[model_type] = metrics
            
            # Ensure model training is complete before moving to next model
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        except Exception as e:
            if args.local_rank in [-1, 0]:
                logger.error(f"Error processing {model_type}: {str(e)}")
            continue
    
    # Only master process handles plotting and saving results
    if args.local_rank in [-1, 0]:
        # Plot combined RC curves
        logger.info("Plotting combined RC curves...")
        plot_combined_rc_curves(
            [metrics for metrics in all_metrics.values()],
            list(all_metrics.keys()),
            output_dir / 'CombinedTrainingCurves.png'
        )
        
        # Save all results
        logger.info("Saving combined results...")
        with open(output_dir / 'CombinedTraining.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_metrics = {}
            for name, metrics in all_metrics.items():
                json_metrics[name] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in metrics.items()
                }
            json.dump(json_metrics, f, indent=2)
        
        logger.info("All experiments completed!")
    
    # Final synchronization
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

if __name__ == '__main__':
    main()