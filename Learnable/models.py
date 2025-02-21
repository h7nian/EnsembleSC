import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class BaseSelector(nn.Module):
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class SRMaxSelector(BaseSelector):
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        return probs.max(dim=1)[0]

class SRDoctorSelector(BaseSelector):
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        return 1 - 1 / torch.sum(probs ** 2, dim=1)

class SREntropySelector(BaseSelector):
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        return -entropy  # Negative because lower entropy means higher confidence

class RLGeoMSelector(BaseSelector):
    def __init__(self, base_model=None):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if hasattr(self.base_model, 'head'):
            W = self.base_model.head.weight
        elif hasattr(self.base_model, 'fc'):
            W = self.base_model.fc.weight
        else:
            sorted_logits, _ = torch.sort(logits, dim=1, descending=True)
            return sorted_logits[:, 0] - sorted_logits[:, 1]
            
        W_norms = torch.norm(W, p=2, dim=1)
        normalized_logits = logits / (W_norms + 1e-8).unsqueeze(0)
        sorted_logits, _ = torch.sort(normalized_logits, dim=1, descending=True)
        return sorted_logits[:, 0] - sorted_logits[:, 1]

class RLConfMSelector(BaseSelector):
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        sorted_logits, _ = torch.sort(logits, dim=1, descending=True)
        return sorted_logits[:, 0] - sorted_logits[:, 1]

class EnsembleSC(nn.Module):
    """Modified implementation with two-layer neural network for score combination"""
    def __init__(self, base_model: nn.Module, hidden_size: int = 32):
        super().__init__()
        self.base_model = base_model
        self.selectors = nn.ModuleList([
            SRMaxSelector(),
            SRDoctorSelector(),
            SREntropySelector(),
            RLGeoMSelector(base_model),
            RLConfMSelector()
        ])
        self.selector_names = ['SRMax', 'SRDoctor', 'SREntropy', 'RLGeoM', 'RLConfM']
        num_selectors = len(self.selectors)
        
        # Two-layer neural network for score combination
        self.score_network = nn.Sequential(
            nn.Linear(num_selectors, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize weights
        for layer in self.score_network:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
    
    def normalize_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Normalize scores to range [-1, 1] per selector
        Args:
            scores: Tensor of shape (num_selectors, batch_size)
        Returns:
            Normalized scores in range [-1, 1]
        """
        min_vals = scores.min(dim=1, keepdim=True)[0]
        max_vals = scores.max(dim=1, keepdim=True)[0]
        
        denom = max_vals - min_vals
        denom[denom == 0] = 1.0
        
        normalized = 2 * (scores - min_vals) / denom - 1
        return normalized
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            logits = self.base_model(x)
        
        selector_scores = []
        for selector in self.selectors:
            scores = selector(logits)
            selector_scores.append(scores)
        
        # Stack and normalize scores
        selector_scores = torch.stack(selector_scores)  # Shape: (num_selectors, batch_size)
        normalized_scores = self.normalize_scores(selector_scores)
        
        # Transpose to (batch_size, num_selectors) for the neural network
        normalized_scores = normalized_scores.t()
        
        # Pass through two-layer network and squeeze output
        combined_scores = self.score_network(normalized_scores).squeeze(-1)
        
        return logits, combined_scores

    def predict_selective(self, x: torch.Tensor, threshold: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, scores = self.forward(x)
        predictions = logits.argmax(dim=1)
        selected = scores >= threshold
        return predictions, scores, selected
    
    def get_selector_importance(self):
        """
        Calculate importance scores for each selector based on the first layer weights
        """
        first_layer = self.score_network[0]
        importance = torch.abs(first_layer.weight).mean(dim=0)
        normalized_importance = F.softmax(importance, dim=0)
        return list(zip(self.selector_names, normalized_importance.detach().cpu().numpy()))

class EnsembleSCLearnable(nn.Module):
    """Modified implementation with two-layer neural network and learnable thresholds"""
    def __init__(self, base_model: nn.Module, hidden_size: int = 32):
        super().__init__()
        self.base_model = base_model
        self.selectors = nn.ModuleList([
            SRMaxSelector(),
            SRDoctorSelector(),
            SREntropySelector(),
            RLGeoMSelector(base_model),
            RLConfMSelector()
        ])
        self.selector_names = ['SRMax', 'SRDoctor', 'SREntropy', 'RLGeoM', 'RLConfM']
        num_selectors = len(self.selectors)
        
        # Two-layer neural network for score combination
        self.score_network = nn.Sequential(
            nn.Linear(num_selectors, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Learnable thresholds
        self.selector_thresholds = nn.Parameter(torch.zeros(num_selectors))
        
        # Initialize weights
        for layer in self.score_network:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        torch.nn.init.zeros_(self.selector_thresholds)
    
    def normalize_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Normalize scores to range [-1, 1] per selector
        Args:
            scores: Tensor of shape (batch_size, num_selectors)
        Returns:
            Normalized scores in range [-1, 1]
        """
        min_vals = scores.min(dim=0, keepdim=True)[0]
        max_vals = scores.max(dim=0, keepdim=True)[0]
        
        denom = max_vals - min_vals
        denom[denom == 0] = 1.0
        
        normalized = 2 * (scores - min_vals) / denom - 1
        return normalized
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            logits = self.base_model(x)
        
        selector_scores = []
        for selector in self.selectors:
            score = selector(logits)
            selector_scores.append(score)
        
        # Stack and normalize scores
        selector_scores = torch.stack(selector_scores, dim=1)  # Shape: (batch_size, num_selectors)
        normalized_scores = self.normalize_scores(selector_scores)
        
        # Apply thresholds and get indicators
        indicators = torch.tanh(normalized_scores - self.selector_thresholds)
        
        # Pass through two-layer network and squeeze output
        ensemble_score = self.score_network(indicators).squeeze(-1)
        
        return logits, ensemble_score

    def predict_selective(self, x: torch.Tensor, threshold: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, scores = self.forward(x)
        predictions = logits.argmax(dim=1)
        selected = scores >= threshold
        return predictions, scores, selected

    def get_selector_importance(self):
        """
        Calculate importance scores for each selector based on the first layer weights
        """
        first_layer = self.score_network[0]
        importance = torch.abs(first_layer.weight).mean(dim=0)
        normalized_importance = F.softmax(importance, dim=0)
        return list(zip(self.selector_names, normalized_importance.detach().cpu().numpy()))

def create_selective_network(model_name: str, model_type: str = "EnsembleSC") -> nn.Module:
    """Create a selective network with specified type."""
    logger.info(f"Loading pretrained model: {model_name}")
    base_model = timm.create_model(model_name, pretrained=True)
    base_model.eval()
    
    for param in base_model.parameters():
        param.requires_grad = False
    
    if model_type == "EnsembleSC":
        return EnsembleSC(base_model)
    elif model_type == "EnsembleSC-Learnable":
        return EnsembleSCLearnable(base_model)
    else:
        raise ValueError(f"Unknown model type: {model_type}")