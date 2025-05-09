from typing import Any

import torch

from .base import BaseSampler


class XTCSampler(BaseSampler):
    """
    XTC (Exclude Top Choices) sampling "turns truncation on its head."
    Instead of pruning low-probability tokens, XTC targets the most probable ones
    under certain conditions to enhance creativity.
    """

    def __init__(
        self,
        top_k: int = 5,
        exclusion_threshold: float = 0.3,
        min_probability: float = 0.1,
    ):
        """
        Initialize the XTC sampler.

        Args:
            top_k: Number of top tokens to potentially exclude
            exclusion_threshold: Probability threshold for exclusion
            min_probability: Minimum probability to consider for exclusion
        """
        super().__init__()
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0 < exclusion_threshold < 1:
            raise ValueError("exclusion_threshold must be in (0, 1)")
        if not 0 < min_probability < 1:
            raise ValueError("min_probability must be in (0, 1)")

        self.top_k = top_k
        self.exclusion_threshold = exclusion_threshold
        self.min_probability = min_probability

    def _apply_sampling(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply XTC filtering to the logits.

        Args:
            logits: Raw logits from the model

        Returns:
            torch.Tensor: XTC filtered logits
        """
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Get top-k probabilities and indices
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)

        # Create a mask for tokens to exclude
        # Exclude tokens that are both in top-k and above the exclusion threshold
        exclusion_mask = torch.zeros_like(probs, dtype=torch.bool)
        for i in range(self.top_k):
            mask = (top_k_probs[:, i] > self.exclusion_threshold) & (
                top_k_probs[:, i] > self.min_probability
            )
            exclusion_mask.scatter_(1, top_k_indices[:, i : i + 1], mask.unsqueeze(1))

        # Set excluded tokens to negative infinity
        filtered_logits = logits.clone()
        filtered_logits[exclusion_mask] = float("-inf")

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
        Generate text using XTC sampling.

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
