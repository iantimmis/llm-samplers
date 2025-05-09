from typing import Any

import torch

from .base import BaseSampler


class MinPSampler(BaseSampler):
    """
    Min-P sampling dynamically adjusts the sampling pool size based on
    the probability of the most likely token.
    """

    def __init__(self, min_p: float = 0.05):
        """
        Initialize the Min-P sampler.

        Args:
            min_p: Minimum probability threshold for tokens
        """
        super().__init__()
        if not 0 < min_p < 1:
            raise ValueError("min_p must be in (0, 1)")
        self.min_p = min_p

    def _apply_sampling(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply Min-P filtering to the logits.

        Args:
            logits: Raw logits from the model

        Returns:
            torch.Tensor: Min-P filtered logits
        """
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Get the maximum probability for each sequence
        max_probs = torch.max(probs, dim=-1, keepdim=True)[0]

        # Calculate the minimum probability threshold
        min_prob_threshold = max_probs * self.min_p

        # Create a mask for tokens to keep
        # Keep tokens with probability above the minimum threshold
        indices_to_keep = probs >= min_prob_threshold

        # Set the filtered logits to negative infinity for tokens below threshold
        filtered_logits = logits.clone()
        filtered_logits[~indices_to_keep] = float("-inf")

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
        Generate text using Min-P sampling.

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
