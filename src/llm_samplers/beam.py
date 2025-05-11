from typing import Any

import torch

from .base import BaseSampler


class BeamSearchSampler(BaseSampler):
    """Beam search sampler that maintains the top k most promising sequences."""

    def __init__(self, beam_width: int = 5):
        """
        Initialize the beam search sampler.

        Args:
            beam_width: Number of sequences to keep at each step
        """
        super().__init__()
        self.beam_width = beam_width

    def sample(
        self,
        model: Any,
        input_ids: torch.Tensor,
        max_length: int = 100,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """
        Sample from the model using beam search.

        Args:
            model: The language model to sample from
            input_ids: Input token IDs
            max_length: Maximum length of the generated sequence
            num_return_sequences: Number of sequences to return
            **kwargs: Additional arguments specific to the sampler

        Returns:
            torch.Tensor: Generated token IDs
        """
        batch_size = input_ids.shape[0]
        sequences = input_ids.clone()
        scores = torch.zeros(batch_size, device=self.device)

        for _ in range(max_length - input_ids.shape[1]):
            # Get logits for all sequences
            logits = self._get_logits(model, sequences)

            # Get top k tokens for each sequence
            probs = torch.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, k=self.beam_width, dim=-1)

            # Expand sequences and scores
            expanded_sequences = sequences.unsqueeze(1).expand(
                batch_size, self.beam_width, sequences.shape[-1]
            )
            expanded_scores = scores.unsqueeze(1).expand(batch_size, self.beam_width)

            # Add new tokens and update scores
            new_sequences = torch.cat(
                [expanded_sequences, topk_indices.unsqueeze(-1)], dim=-1
            )
            new_scores = expanded_scores + torch.log(topk_probs)

            # Reshape to combine all candidates
            new_sequences = new_sequences.reshape(
                batch_size, -1, new_sequences.shape[-1]
            )
            new_scores = new_scores.reshape(batch_size, -1)

            # Select top k sequences
            topk_scores, topk_indices = torch.topk(
                new_scores, k=self.beam_width, dim=-1
            )
            sequences = torch.gather(
                new_sequences,
                1,
                topk_indices.unsqueeze(-1).expand(-1, -1, new_sequences.shape[-1]),
            )
            scores = topk_scores

        # Return the best sequences
        return sequences[:, :num_return_sequences, :]
