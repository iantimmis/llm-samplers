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
        # Make sure num_return_sequences doesn't exceed beam_width
        num_return_sequences = min(num_return_sequences, self.beam_width)

        # Clone original input_ids
        sequences = input_ids.clone()
        seq_length = sequences.shape[1]
        scores = torch.zeros(batch_size, 1, device=self.device)

        # First step: expand to beam_width candidates
        if seq_length < max_length:
            logits = self._get_logits(model, sequences)
            probs = torch.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, k=self.beam_width, dim=-1)

            # Create beam_width copies of each sequence
            sequences = sequences.unsqueeze(1).expand(
                batch_size, self.beam_width, seq_length
            )

            # Add the new token to each sequence
            topk_indices = topk_indices.unsqueeze(-1)
            sequences = torch.cat([sequences, topk_indices], dim=2)

            # Update scores
            scores = scores.expand(batch_size, self.beam_width)
            scores = scores + torch.log(topk_probs)

            # Continue generating
            for _ in range(max_length - seq_length - 1):
                curr_len = sequences.shape[2]
                # Reshape for model input - from [batch, beam, seq] to [batch*beam, seq]
                flat_sequences = sequences.reshape(
                    batch_size * self.beam_width, curr_len
                )

                # Get logits for all beam sequences
                logits = self._get_logits(model, flat_sequences)
                # Reshape logits to [batch, beam, vocab]
                logits = logits.reshape(batch_size, self.beam_width, -1)

                # Calculate probabilities and get top k for each beam
                probs = torch.softmax(logits, dim=-1)
                # For each sequence in each batch, get beam_width top probabilities
                topk_probs, topk_indices = torch.topk(probs, k=self.beam_width, dim=-1)

                # Calculate combined scores for all possible extensions
                # [batch, beam, beam]
                beam_scores = scores.unsqueeze(-1) + torch.log(topk_probs)
                # Reshape to [batch, beam*beam]
                beam_scores = beam_scores.reshape(batch_size, -1)

                # Get the top beam_width scores and their indices
                topk_beam_scores, topk_beam_indices = torch.topk(
                    beam_scores, k=self.beam_width, dim=-1
                )

                # Convert flat indices to beam indices and token indices
                beam_indices = topk_beam_indices // self.beam_width
                token_indices = topk_beam_indices % self.beam_width

                # Gather the best beam sequences
                new_sequences = []
                for batch_idx in range(batch_size):
                    batch_new_sequences = []
                    for beam_idx in range(self.beam_width):
                        # Get the best beam for this position
                        best_beam = beam_indices[batch_idx, beam_idx]
                        # Get the corresponding token
                        best_token = topk_indices[
                            batch_idx, best_beam, token_indices[batch_idx, beam_idx]
                        ]
                        # Get the sequence for the best beam
                        seq = sequences[batch_idx, best_beam].clone()
                        # Add the new token
                        batch_new_sequences.append(
                            torch.cat([seq, best_token.unsqueeze(0)], dim=0)
                        )
                    new_sequences.append(torch.stack(batch_new_sequences))

                # Update sequences and scores
                sequences = torch.stack(new_sequences)
                scores = topk_beam_scores

        # Pad sequences to max_length if needed
        if sequences.shape[2] < max_length:
            padding = torch.zeros(
                batch_size,
                self.beam_width,
                max_length - sequences.shape[2],
                dtype=sequences.dtype,
                device=sequences.device,
            )
            sequences = torch.cat([sequences, padding], dim=2)

        # Return the top num_return_sequences sequences for each batch
        return sequences[:, :num_return_sequences, :]
