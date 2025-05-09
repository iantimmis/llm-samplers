from typing import Any, List, Optional, Set

import torch

from .base import BaseSampler


class AntiSlopSampler(BaseSampler):
    """
    Anti-Slop sampling down-weights probabilities at word & phrase level.
    Uses backtracking to retry with adjusted token probabilities if it
    encounters a disallowed word/phrase.
    """

    def __init__(
        self,
        disallowed_tokens: Optional[Set[int]] = None,
        disallowed_phrases: Optional[List[List[int]]] = None,
        penalty: float = 0.5,
        max_retries: int = 3,
    ):
        """
        Initialize the Anti-Slop sampler.

        Args:
            disallowed_tokens: Set of token IDs to penalize
            disallowed_phrases: List of token ID sequences to penalize
            penalty: Probability penalty factor (0-1)
            max_retries: Maximum number of retries for backtracking
        """
        super().__init__()
        if not 0 <= penalty <= 1:
            raise ValueError("penalty must be between 0 and 1")
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")

        self.disallowed_tokens = disallowed_tokens or set()
        self.disallowed_phrases = disallowed_phrases or []
        self.penalty = penalty
        self.max_retries = max_retries

    def _apply_sampling(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply Anti-Slop filtering to the logits.

        Args:
            logits: Raw logits from the model

        Returns:
            torch.Tensor: Anti-Slop filtered logits
        """
        filtered_logits = logits.clone()

        # Apply penalty to disallowed tokens
        if self.disallowed_tokens:
            for token_id in self.disallowed_tokens:
                filtered_logits[:, token_id] *= self.penalty

        return filtered_logits

    def _check_phrases(self, generated: torch.Tensor) -> bool:
        """
        Check if any disallowed phrases are present in the generated sequence.

        Args:
            generated: Generated token sequence

        Returns:
            bool: True if a disallowed phrase is found
        """
        for phrase in self.disallowed_phrases:
            phrase_len = len(phrase)
            for i in range(generated.shape[1] - phrase_len + 1):
                if torch.all(
                    generated[:, i : i + phrase_len]
                    == torch.tensor(phrase, device=generated.device)
                ):
                    return True
        return False

    def sample(
        self,
        model: Any,
        input_ids: torch.Tensor,
        max_length: int = 100,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate text using Anti-Slop sampling.

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
        retry_count = 0

        while retry_count < self.max_retries:
            current_generated = generated.clone()

            for _ in range(max_length):
                logits = self._get_logits(model, current_generated)
                logits = self._apply_sampling(logits)
                next_tokens = self._sample_from_logits(logits, num_samples=1)

                current_generated = torch.cat([current_generated, next_tokens], dim=1)

                # Check for disallowed phrases
                if self._check_phrases(current_generated):
                    retry_count += 1
                    break

                # Check if all sequences have generated an EOS token
                if (next_tokens == model.config.eos_token_id).any():
                    return current_generated

            # If we've reached max_length without finding a disallowed phrase
            if not self._check_phrases(current_generated):
                return current_generated

        # If we've exhausted all retries, return the last generated sequence
        return current_generated
