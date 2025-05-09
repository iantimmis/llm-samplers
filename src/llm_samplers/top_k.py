from typing import Any

import torch

from .base import BaseSampler


class TopKSampler(BaseSampler):
    """
    Top-K sampling considers only the 'k' most probable tokens.
    All other tokens are set to have zero probability.
    """

    def __init__(self, k: int = 50):
        """
        Initialize the Top-K sampler.

        Args:
            k: Number of top tokens to consider
        """
        super().__init__()
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k

    def _apply_sampling(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply Top-K filtering to the logits.

        Args:
            logits: Raw logits from the model

        Returns:
            torch.Tensor: Top-K filtered logits
        """
        vocab_size = logits.size(-1)
        k = min(self.k, vocab_size)  # Ensure k doesn't exceed vocabulary size

        # Get the top k values and indices
        top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)

        # Create a new tensor of zeros with the same shape as logits
        filtered_logits = torch.full_like(logits, float("-inf"))

        # Set the top k values in their original positions
        filtered_logits.scatter_(1, top_k_indices, top_k_values)

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
        Generate text using Top-K sampling.

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
