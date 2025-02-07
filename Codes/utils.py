import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import logging
import json
from pathlib import Path

def setup_logging(log_dir: str = 'logs') -> None:
    """Setup logging configuration"""
    Path(log_dir).mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/training.log'),
            logging.StreamHandler()
        ]
    )

def plot_rc_curve(metrics: Dict[str, List[float]], 
                  save_path: str = None) -> None:
    """Plot risk-coverage curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['coverages'], metrics['risks'])
    plt.xlabel('Coverage')
    plt.ylabel('Risk')
    plt.title('Risk-Coverage Curve')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def calculate_aurc(coverages: List[float], 
                  risks: List[float],
                  alpha: float = None) -> float:
    """Calculate area under risk-coverage curve, optionally up to coverage alpha"""
    if alpha is not None:
        # Find index closest to alpha coverage
        idx = min(range(len(coverages)), 
                 key=lambda i: abs(coverages[i] - alpha))
        coverages = coverages[:idx+1]
        risks = risks[:idx+1]
    
    # Calculate area using trapezoidal rule
    aurc = np.trapz(y=risks, x=coverages)
    
    if alpha is not None:
        # Normalize by alpha
        aurc /= alpha
        
    return aurc

def save_metrics(metrics: Dict[str, List[float]], 
                save_path: str) -> None:
    """Save metrics to JSON file"""
    # Convert numpy arrays to lists for JSON serialization
    metrics_json = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in metrics.items()
    }
    
    with open(save_path, 'w') as f:
        json.dump(metrics_json, f, indent=4)

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count