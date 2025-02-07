import argparse
from pathlib import Path
import torch
import logging

from dataprocess import ImageNetDataset
from models import create_selective_network
from train import SelectiveTrainer
from utils import setup_logging, plot_rc_curve, calculate_aurc, save_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Train Selective Classification')
    parser.add_argument('--data-root', type=str, default='ImageNet/',
                      help='Path to ImageNet dataset')
    parser.add_argument('--model-name', type=str, 
                      default='eva_giant_patch14_224.clip_ft_in1k',
                      help='Name of the pretrained model')
    parser.add_argument('--output-dir', type=str, default='outputs',
                      help='Directory to save outputs')
    parser.add_argument('--batch-size', type=int, default=1024,
                      help='Batch size')
    parser.add_argument('--cal-size', type=float, default=0.2,
                      help='Proportion of data for calibration')
    parser.add_argument('--target-coverage', type=float, default=0.8,
                      help='Target coverage rate')
    parser.add_argument('--n-epochs', type=int, default=10,
                      help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate') 
    parser.add_argument('--num-workers', type=int, default=2,
                      help='Number of data loading workers')
    
    return parser.parse_args()

def main():
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
        device=args.device
    )
    
    # Train model
    logger.info("Starting training...")
    training_history = trainer.train(
        train_loader=dataloaders['calibration'],
        val_loader=dataloaders['validation'],
        n_epochs=args.n_epochs
    )
    
    # Final evaluation
    logger.info("Final evaluation...")
    final_metrics = trainer.evaluate(dataloaders['validation'])
    
    # Calculate AURC
    aurc = calculate_aurc(final_metrics['coverages'], final_metrics['risks'])
    aurc_90 = calculate_aurc(final_metrics['coverages'], 
                            final_metrics['risks'], 
                            alpha=0.9)
    logger.info(f"Final AURC: {aurc:.4f}")
    logger.info(f"Final AURC@90: {aurc_90:.4f}")
    
    # Save results
    logger.info("Saving results...")
    save_metrics(final_metrics, output_dir / 'metrics.json')
    plot_rc_curve(final_metrics, output_dir / 'rc_curve.png')
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'metrics': final_metrics,
        'args': vars(args)
    }, output_dir / 'model.pt')
    
    logger.info("Done!")

if __name__ == '__main__':
    main()