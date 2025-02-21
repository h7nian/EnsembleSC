import logging
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple

def setup_logging(output_dir: str, local_rank: int = -1) -> None:
    """Setup logging configuration."""
    # Only setup logging for master process
    if local_rank in [-1, 0]:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Path(output_dir) / 'train.log'),
                logging.StreamHandler()
            ]
        )

def calculate_rc_curve(scores: List[float], correct: List[bool], 
                      num_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate risk-coverage curve using selective risk.
    
    Args:
        scores: Confidence scores from selector
        correct: Binary array indicating whether predictions are correct
        num_points: Number of points on the curve
        
    Returns:
        coverages: Array of coverage values
        selective_risks: Array of selective risk values
        thresholds: Array of threshold values
    """
    # Convert to numpy arrays
    scores = np.array(scores)
    correct = np.array(correct)
    
    # Sort by confidence scores
    sorted_indices = np.argsort(scores)[::-1]  # Descending order
    sorted_scores = scores[sorted_indices]
    sorted_correct = correct[sorted_indices]
    
    # Calculate coverage and selective risk points
    coverages = []
    selective_risks = []
    thresholds = []
    
    total_samples = len(scores)
    
    # Generate evenly spaced coverage points
    for i in range(num_points + 1):
        coverage = i / num_points
        n_samples = int(coverage * total_samples)
        
        if n_samples == 0:
            continue
            
        # Calculate selective risk
        selected_correct = sorted_correct[:n_samples]
        errors = np.logical_not(selected_correct)
        
        selective_risk = np.sum(errors) / n_samples
        
        coverages.append(coverage)
        selective_risks.append(selective_risk)
        
        # Store threshold
        if n_samples < len(scores):
            thresholds.append(sorted_scores[n_samples - 1])
        else:
            thresholds.append(sorted_scores[-1])
    
    return np.array(coverages), np.array(selective_risks), np.array(thresholds)

def plot_combined_rc_curves(metrics_list: List[dict], names: List[str], output_path: Path):
    """Plot RC curves for all models and base selectors"""
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    # Plot each curve
    for idx, (metrics, name) in enumerate(zip(metrics_list, names)):
        if isinstance(metrics['scores'], dict):  # Base selectors
            for selector_idx, (selector_name, selector_metrics) in enumerate(metrics.items()):
                if 'coverages' in selector_metrics and 'risks' in selector_metrics:
                    coverages, risks = selector_metrics['coverages'], selector_metrics['risks']
                else:
                    coverages, risks, _ = calculate_rc_curve(
                        selector_metrics['scores'], 
                        selector_metrics['correct']
                    )
                
                plt.plot(coverages, risks,
                        label=f"{name}-{selector_name}",
                        color=colors[selector_idx % 20],
                        marker='o', markersize=4, markevery=0.1)
        else:  # Ensemble models
            if 'coverages' in metrics and 'risks' in metrics:
                coverages, risks = metrics['coverages'], metrics['risks']
            else:
                coverages, risks, _ = calculate_rc_curve(
                    metrics['scores'],
                    metrics['correct']
                )
            
            plt.plot(coverages, risks,
                    label=name,
                    color=colors[(idx + 10) % 20], 
                    marker='s', markersize=4, markevery=0.1)
    
    plt.grid(True)
    plt.xlabel('Coverage')
    plt.ylabel('Selective Risk')
    plt.title('Risk-Coverage Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
def evaluate_at_coverage(coverages: np.ndarray, risks: np.ndarray, 
                       thresholds: np.ndarray, target_coverage: float) -> Dict[str, float]:
    """
    Evaluate metrics at a specific target coverage point.
    """
    idx = np.argmin(np.abs(coverages - target_coverage))
    return {
        'threshold': thresholds[idx],
        'risk': risks[idx],
        'coverage': coverages[idx]
    }    

def save_metrics(metrics: dict, output_path: Path):
    """Save metrics to JSON file"""
    # Convert numpy arrays to lists for JSON serialization
    metrics_json = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics_json[key] = value.tolist()
        elif isinstance(value, dict):
            metrics_json[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                               for k, v in value.items()}
        else:
            metrics_json[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)