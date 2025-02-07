import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class SelectiveNetwork(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.selectors = nn.ModuleList([
            SRMaxSelector(),
            SRDoctorSelector(),
            SREntropySelector(),
            RLGeoMSelector(base_model),  # Pass base_model here
            RLConfMSelector()
        ])
        
        # Learnable parameters
        self.selector_weights = nn.Parameter(torch.zeros(len(self.selectors)))
        self.selector_thresholds = nn.Parameter(torch.zeros(len(self.selectors)))
        self.ensemble_threshold = nn.Parameter(torch.tensor(0.0))
        
        # Initialize weights with small random values
        torch.nn.init.normal_(self.selector_weights, mean=0.0, std=0.01)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get base model logits
        with torch.set_grad_enabled(False):  # No gradient needed for base model
            logits = self.base_model(x)
        
        # Apply each selector
        selector_outputs = []
        for selector, threshold in zip(self.selectors, self.selector_thresholds):
            score = selector(logits)
            # Soft indicator function using tanh
            indicator = torch.tanh(score - threshold)
            selector_outputs.append(indicator)
            
        # Stack all selector outputs
        selector_outputs = torch.stack(selector_outputs, dim=1)  # [B, num_selectors]
        
        # Weighted combination of selectors
        ensemble_score = torch.sum(self.selector_weights * selector_outputs, dim=1)
        
        # Final selection score
        selection_score = torch.tanh(ensemble_score - self.ensemble_threshold)
        
        return logits, selection_score
    
    def predict_with_selection(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, selection_score = self.forward(x)
        predictions = logits.argmax(dim=1)
        selected = selection_score > 0
        return predictions, selected, selection_score

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
    """Geometric margin-based selector using normalized logits"""
    def __init__(self, base_model=None):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if hasattr(self.base_model, 'head'):  # For EVA model
            W = self.base_model.head.weight
        elif hasattr(self.base_model, 'fc'):  # For ResNet
            W = self.base_model.fc.weight
        else:
            # If can't get weights, fallback to regular confidence margin
            sorted_logits, _ = torch.sort(logits, dim=1, descending=True)
            return sorted_logits[:, 0] - sorted_logits[:, 1]
            
        # Normalize logits by the L2 norm of corresponding weight vectors
        W_norms = torch.norm(W, p=2, dim=1)  # [num_classes]
        normalized_logits = logits / (W_norms + 1e-8).unsqueeze(0)  # [batch_size, num_classes]
        
        # Get top 2 normalized logits
        sorted_logits, _ = torch.sort(normalized_logits, dim=1, descending=True)
        return sorted_logits[:, 0] - sorted_logits[:, 1]

class RLConfMSelector(BaseSelector):
    """Confidence margin-based selector using raw logits"""
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Directly use raw logits difference
        sorted_logits, _ = torch.sort(logits, dim=1, descending=True)
        return sorted_logits[:, 0] - sorted_logits[:, 1]

def create_selective_network(model_name: str) -> SelectiveNetwork:
    """Create a selective network with a pretrained base model."""
    # Load pretrained model using timm
    logger.info(f"Loading pretrained model: {model_name}")
    base_model = timm.create_model(model_name, pretrained=True)
    base_model.eval()  # Set to evaluation mode
    
    # Freeze base model parameters
    for param in base_model.parameters():
        param.requires_grad = False
        
    return SelectiveNetwork(base_model)