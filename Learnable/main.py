import argparse
from pathlib import Path
import torch
import logging
from dataprocess import ImageNetDataset
from models import create_selective_network
from train import SelectiveTrainer
from utils import setup_logging, plot_rc_curves, calculate_aurc, save_metrics, plot_all_rc_curves
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Train Selective Classification')
    
    # Dataset related arguments
    parser.add_argument('--data-root', type=str, default='/users/0/zhan9381/EnsembleSC/Codes/ImageNet',
                      help='Path to ImageNet dataset')
    parser.add_argument('--cal-size', type=float, default=0.2,
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
    parser.add_argument('--n-epochs', type=int, default=1,
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
    parser.add_argument('--experiment-name', type=str, default='EnsembleSC',
                      help='Name of the experiment for organizing outputs')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process the arguments
    # Create full output directory path including experiment name
    args.output_dir = os.path.join(args.output_dir, args.experiment_name)
    
    # If cache_dir is specified, create full cache directory path
    if args.cache_dir is not None:
        args.cache_dir = os.path.join(args.cache_dir, args.experiment_name)
        # Create cache directory if it doesn't exist
        os.makedirs(args.cache_dir, exist_ok=True)
    
    return args

def main():
    # Parse arguments
    args = parse_args()
    
    # Whether to use GPU
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(str(output_dir))
    logger = logging.getLogger(__name__)
    
    # Log arguments
    logger.info(f"Arguments: {args}")
    logger.info(f"Device: {args.device}")
    if args.cache_dir:
        logger.info(f"Cache directory: {args.cache_dir}")
    
    # Create datasets and dataloaders
    logger.info("Creating datasets...")
    dataset = ImageNetDataset(args)
    
    # Create model first since we need it for calibration set
    logger.info(f"Creating model: {args.model_name}")
    model = create_selective_network(args.model_name)
    
    # Get dataloaders
    logger.info("Creating dataloader...")
    dataloaders = dataset.get_dataloaders(base_model=model.base_model)
    
    # Create trainer
    trainer = SelectiveTrainer(
        model=model,
        learning_rate=args.lr,
        device=args.device,
        val_interval=args.val_interval
    )
    
    # Train model
    logger.info("Starting training...")
    train_metrics = trainer.train(
        train_loader=dataloaders['calibration'],
        val_loader=dataloaders['validation'],
        n_epochs=args.n_epochs
    )
    
    # Final evaluation
    logger.info("Final evaluation...")
    final_metrics = trainer.evaluate(dataloaders['validation'])
    
    # Plot Curves and save results
    logger.info("\nSaving results...")
    save_metrics(final_metrics, output_dir / 'metrics.json')
    plot_all_rc_curves(final_metrics, output_dir / 'all_rc_curves.png')
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'metrics': final_metrics,
        'args': vars(args)
    }, output_dir / 'model.pt')
    
    logger.info("Done!")

if __name__ == '__main__':
    main()