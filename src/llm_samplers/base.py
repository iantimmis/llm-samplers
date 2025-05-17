from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseSampler(ABC):
    """Base class for all samplers."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def sample(
        self,
        model: Any,
        input_ids: torch.Tensor,
        max_length: int = 100,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """
        Sample from the model using the specific sampling strategy.

        Args:
            model: The language model to sample from
            input_ids: Input token IDs
            max_length: Maximum length of the generated sequence
            num_return_sequences: Number of sequences to return
            **kwargs: Additional arguments specific to the sampler

        Returns:
            torch.Tensor: Generated token IDs
        """
        pass

    def _get_logits(self, model: Any, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get the logits from the model for the given input.

        Args:
            model: The language model
            input_ids: Input token IDs

        Returns:
            torch.Tensor: Logits from the model
        """
        # Ensure input_ids are on the correct device
        input_ids = input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]
            
        # Ensure logits are on the correct device
        return logits.to(self.device)

    def _apply_sampling(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply the sampling strategy to the logits.

        Args:
            logits: Raw logits from the model

        Returns:
            torch.Tensor: Processed logits ready for sampling
        """
        # Ensure logits are on the correct device
        return logits.to(self.device)

    def _sample_from_logits(
        self, logits: torch.Tensor, num_samples: int = 1
    ) -> torch.Tensor:
        """
        Sample from the processed logits.

        Args:
            logits: Processed logits
            num_samples: Number of samples to generate

        Returns:
            torch.Tensor: Sampled token IDs
        """
        # Ensure logits are on the correct device
        logits = logits.to(self.device)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=num_samples)
