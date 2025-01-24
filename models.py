import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod
import timm

# Base selector and individual selectors from previous implementation
class BaseSelector(ABC):
    @abstractmethod
    def get_score(self, logits: torch.Tensor) -> torch.Tensor:
        pass
    def __repr__(self) -> str:
        return self.__class__.__name__

# Selectors implementation (same as before)
class SRMaxSelector(BaseSelector):
    def get_score(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        return probs.max(dim=1)[0]

class SRDoctorSelector(BaseSelector):
    def get_score(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        return 1 - 1 / torch.norm(probs, p=2, dim=1) ** 2

class SREntropySelector(BaseSelector):
    def get_score(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        return -torch.sum(probs * torch.log(probs + 1e-10), dim=1)

class RLGeoMSelector(BaseSelector):
    """Geometric margin-based selector using normalized logits"""
    def __init__(self, base_model=None):
        super().__init__()
        self.base_model = base_model
        
    def get_score(self, logits: torch.Tensor) -> torch.Tensor:
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
    def get_score(self, logits: torch.Tensor) -> torch.Tensor:
        # Directly use raw logits difference
        sorted_logits, _ = torch.sort(logits, dim=1, descending=True)
        return sorted_logits[:, 0] - sorted_logits[:, 1]
class SelectiveNetwork(nn.Module):
    """Network combining base model with selective classification"""
    def __init__(self, model_name: str = 'eva_giant_patch14_224.clip_ft_in1k'):
        super().__init__()
        # Load pretrained model
        self.base_model = timm.create_model(model_name, pretrained=True)
        self.base_model.eval()  # Set to evaluation mode
        
        # Initialize selectors
        self.selectors = [
                SRMaxSelector(),
                SRDoctorSelector(),
                SREntropySelector(),
                RLGeoMSelector(base_model=self.base_model),
                RLConfMSelector()
            ]
        
        # Learnable weights for combining selectors
        self.selector_weights = nn.Parameter(
            torch.ones(len(self.selectors)) / len(self.selectors)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both logits and selection scores
        """
        with torch.no_grad():
            logits = self.base_model(x)
        
        # Get scores from all selectors
        selector_scores = []
        for selector in self.selectors:
            scores = selector.get_score(logits)
            # Normalize scores
            scores = (scores - scores.mean()) / (scores.std() + 1e-8)
            selector_scores.append(scores)
            
        selector_scores = torch.stack(selector_scores)
        
        # Apply softmax to weights
        weights = F.softmax(self.selector_weights, dim=0)
        
        # Combine scores
        combined_scores = torch.sum(
            weights.view(-1, 1) * selector_scores, 
            dim=0
        )
        
        return logits, combined_scores
    
    def predict_selective(self, 
                         x: torch.Tensor,
                         threshold: Optional[float] = None,
                         coverage: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make selective predictions with either threshold or coverage
        Returns: predictions, scores, mask of selected samples
        """
        assert (threshold is None) != (coverage is None), \
            "Must specify either threshold or coverage"
            
        logits, scores = self.forward(x)
        predictions = logits.argmax(dim=1)
        
        if coverage is not None:
            # Sort scores to find threshold for desired coverage
            sorted_scores, _ = torch.sort(scores, descending=True)
            k = int(len(scores) * coverage)
            threshold = sorted_scores[k-1]
            
        # Get selection mask
        selected = scores >= threshold
        
        return predictions, scores, selected

def create_selective_network(model_name: str = 'eva_giant_patch14_224.clip_ft_in1k') -> SelectiveNetwork:
    """Helper function to create selective network"""
    return SelectiveNetwork(model_name)