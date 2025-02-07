import logging
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from sklearn.metrics import auc

def setup_logging(output_dir: str) -> None:
    """Setup logging configuration."""
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
        errors = np.logical_not(selected_correct)  # Convert to error indicators
        
        # Selective Risk = E[loss * g(x)] / coverage
        # Here g(x) is 1 for selected samples, 0 otherwise
        # loss is 1 for errors, 0 for correct predictions
        selective_risk = np.sum(errors) / n_samples  # This is E[loss * g(x)] / coverage
        
        coverages.append(coverage)
        selective_risks.append(selective_risk)
        
        # Store threshold
        if n_samples < len(scores):
            thresholds.append(sorted_scores[n_samples - 1])
        else:
            thresholds.append(sorted_scores[-1])
    
    return np.array(coverages), np.array(selective_risks), np.array(thresholds)

def calculate_aurc(coverages: np.ndarray, risks: np.ndarray) -> float:
    """Calculate Area Under Risk-Coverage curve."""
    # Remove duplicate coverage points to ensure monotonicity
    unique_coverages, unique_indices = np.unique(coverages, return_index=True)
    unique_risks = risks[unique_indices]
    
    # Calculate AUC
    aurc = auc(unique_coverages, unique_risks)
    return aurc

def plot_rc_curves(metrics: Dict[str, Any], save_path: Path) -> None:
    """Plot Risk-Coverage curve."""
    plt.figure(figsize=(10, 6))
    
    # Calculate and plot RC curve
    coverages, selective_risks, _ = calculate_rc_curve(
        metrics['scores'],
        metrics['correct']
    )
    
    plt.plot(coverages, selective_risks, '-', label='Selective Risk-Coverage curve')
    
    # Calculate and display AURC
    aurc = calculate_aurc(coverages, selective_risks)
    plt.text(0.6, 0.2, f'AURC: {aurc:.4f}', 
             transform=plt.gca().transAxes)
    
    # Add labels and title
    plt.xlabel('Coverage')
    plt.ylabel('Selective Risk')
    plt.title('Selective Risk-Coverage Curve')
    plt.grid(True)
    plt.legend()
    
    # Save plot
    plt.savefig(save_path)
    plt.close()

def plot_all_rc_curves(metrics: Dict[str, Any], save_path: Path) -> None:
    """Plot multiple RC curves for comparison."""
    plt.figure(figsize=(12, 8))
    
    # Plot RC curve for each selector
    selector_names = [
        'SR-Max', 'SR-Doctor', 'SR-Entropy', 
        'RL-GeoM', 'RL-ConfM', 'Ensemble'
    ]
    
    # Define colors and markers for each selector
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    markers = ['o', 's', '^', 'v', 'D', 'p']
    
    for selector_name, color, marker in zip(selector_names, colors, markers):
        if f'{selector_name.lower()}_scores' in metrics:
            coverages, selective_risks, _ = calculate_rc_curve(
                metrics[f'{selector_name.lower()}_scores'],
                metrics['correct']
            )
            plt.plot(coverages, selective_risks, '-', color=color, 
                    label=selector_name, marker=marker, markevery=0.1)
            
            # Calculate and store AURC
            aurc = calculate_aurc(coverages, selective_risks)
            metrics[f'{selector_name.lower()}_aurc'] = aurc
            
            # Add AURC to legend
            plt.text(0.05, 0.95 - 0.05 * selector_names.index(selector_name),
                    f'{selector_name} AURC: {aurc:.4f}',
                    transform=plt.gca().transAxes,
                    color=color)
    
    # Add labels and title
    plt.xlabel('Coverage', fontsize=12)
    plt.ylabel('Selective Risk', fontsize=12)
    plt.title('Selective Risk-Coverage Curves Comparison', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Highlight specific coverage points
    coverage_points = [0.8, 0.9, 0.95]
    for coverage in coverage_points:
        plt.axvline(x=coverage, color='gray', linestyle='--', alpha=0.3)
        
    # Save plot with tight layout
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def save_metrics(metrics: Dict[str, Any], save_path: Path) -> None:
    """Save metrics to JSON file."""
    # Convert numpy arrays and other non-serializable types to lists
    serializable_metrics = {}
    
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()
        elif isinstance(value, np.float32) or isinstance(value, np.float64):
            serializable_metrics[key] = float(value)
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.bool_):
            serializable_metrics[key] = [bool(x) for x in value]
        else:
            serializable_metrics[key] = value
    
    # Save to file
    with open(save_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)

def load_metrics(load_path: Path) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    with open(load_path, 'r') as f:
        metrics = json.load(f)
    return metrics