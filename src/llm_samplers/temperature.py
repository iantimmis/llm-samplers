from typing import Any

import torch

from .base import BaseSampler


class TemperatureSampler(BaseSampler):
    """
    Temperature sampling adjusts the "sharpness" of the probability distribution.

    - Low temperature (<1.0): More deterministic, picks high-probability tokens
    - High temperature (>1.0): More random, flatter distribution
    """

    def __init__(self, temperature: float = 1.0):
        """
        Initialize the temperature sampler.

        Args:
            temperature: Temperature value to control sampling sharpness
        """
        super().__init__()
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        self.temperature = temperature

    def _apply_sampling(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to the logits.

        Args:
            logits: Raw logits from the model

        Returns:
            torch.Tensor: Temperature-scaled logits
        """
        return logits / self.temperature

    def sample(
        self,
        model: Any,
        input_ids: torch.Tensor,
        max_length: int = 100,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate text using temperature sampling.

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
