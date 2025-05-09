from typing import Any

import torch

from .base import BaseSampler


class TopPSampler(BaseSampler):
    """
    Top-P (Nucleus) sampling selects the smallest set of tokens whose
    cumulative probability exceeds threshold 'p'.
    """

    def __init__(self, p: float = 0.9):
        """
        Initialize the Top-P sampler.

        Args:
            p: Probability threshold for nucleus sampling
        """
        super().__init__()
        if not 0 < p <= 1:
            raise ValueError("p must be in (0, 1]")
        self.p = p

    def _apply_sampling(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply Top-P filtering to the logits.

        Args:
            logits: Raw logits from the model

        Returns:
            torch.Tensor: Top-P filtered logits
        """
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        # Calculate cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Create a mask for tokens to keep
        # Keep tokens until cumulative probability exceeds p
        sorted_indices_to_remove = cumulative_probs > self.p

        # Always keep at least one token
        sorted_indices_to_remove[..., 0] = 0

        # Create a boolean mask of the same shape as the input
        indices_to_remove = torch.zeros_like(probs, dtype=torch.bool)
        indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)

        # Set the filtered logits to negative infinity
        filtered_logits = logits.clone()
        filtered_logits[indices_to_remove] = float("-inf")

        return filtered_logits

    def sample(
        self,
        model: Any,
        input_ids: torch.Tensor,
        max_length: int = 100,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate text using Top-P sampling.

        Args:
            model: The language model to sample from
            input_ids: Input token IDs
            max_length: Maximum length of the generated sequence
            num_return_sequences: Number of sequences to return
            **kwargs: Additional arguments

        Returns:
            torch.Tensor: Generated token IDs
        """
        generated = input_ids.clone()

        for _ in range(max_length):
            logits = self._get_logits(model, generated)
            logits = self._apply_sampling(logits)
            next_tokens = self._sample_from_logits(logits, num_samples=1)

            generated = torch.cat([generated, next_tokens], dim=1)

            # Check if all sequences have generated an EOS token
            if (next_tokens == model.config.eos_token_id).any():
                break

        return generated
