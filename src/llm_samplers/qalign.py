from typing import Any

import torch
from torch import nn

from .base import BaseSampler


class QAlignSampler(BaseSampler):
    """QAlign sampler that uses MCMC to improve model outputs based on a reward model.

    This implementation is based on the paper:
    "Sample, Don't Search: Rethinking Test-Time Alignment for Language Models"
    by Faria et al. (2024)
    Paper: https://arxiv.org/abs/2504.03790

    QAlign uses Markov Chain Monte Carlo (MCMC) to align model outputs at test time
    without requiring model fine-tuning. It converges to sampling from the optimal
    aligned distribution as test-time compute scales.
    """

    def __init__(
        self,
        reward_model: nn.Module,
        num_steps: int = 10,
        temperature: float = 1.0,
        beta: float = 1.0,
        proposal_temp: float = 1.0,
    ):
        """Initialize QAlign sampler.

        Args:
            reward_model: The reward model to use for scoring outputs
            num_steps: Number of MCMC steps to perform
            temperature: Temperature for sampling from the base model
            beta: Inverse temperature for the reward model (higher = more emphasis on reward)
            proposal_temp: Temperature for the proposal distribution
        """
        super().__init__()

        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        if proposal_temp <= 0:
            raise ValueError("Proposal temperature must be positive")

        self.reward_model = reward_model
        self.num_steps = num_steps
        self.temperature = temperature
        self.beta = beta
        self.proposal_temp = proposal_temp

    def sample(
        self,
        model: Any,
        input_ids: torch.Tensor,
        max_length: int = 100,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Sample from the model using QAlign.

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
        current_ids = input_ids.clone()

        # Generate initial sequences
        for _ in range(max_length - input_ids.shape[1]):
            logits = self._get_logits(model, current_ids)
            logits = logits / self.temperature
            next_tokens = self._sample_from_logits(logits)

            # Ensure next_tokens has the correct batch dimension
            if next_tokens.shape[0] != batch_size:
                next_tokens = next_tokens.expand(batch_size, -1)

            current_ids = torch.cat([current_ids, next_tokens], dim=1)

            # Stop if all sequences are complete
            if (
                hasattr(model.config, "eos_token_id")
                and (next_tokens == model.config.eos_token_id).all()
            ):
                break

        # Perform MCMC steps
        for _ in range(self.num_steps):
            # Generate proposal
            proposal_ids = self._generate_proposal(model, current_ids)

            # Compute acceptance probability
            current_reward = self._compute_reward(current_ids)
            proposal_reward = self._compute_reward(proposal_ids)

            # Metropolis-Hastings acceptance
            acceptance_prob = torch.exp(
                self.beta * (proposal_reward - current_reward)
            ).clamp(0, 1)

            # Accept or reject
            accept = torch.rand_like(acceptance_prob) < acceptance_prob
            current_ids = torch.where(
                accept.unsqueeze(-1).expand_as(current_ids), proposal_ids, current_ids
            )

        return current_ids

    def _generate_proposal(self, model: Any, current_ids: torch.Tensor) -> torch.Tensor:
        """Generate a proposal sequence using the base model.

        Args:
            model: The language model
            current_ids: Current sequence of token IDs

        Returns:
            torch.Tensor: Proposed sequence of token IDs
        """
        batch_size = current_ids.shape[0]
        seq_len = current_ids.shape[1]

        # Randomly select a position to resample for each example in the batch
        pos = torch.randint(0, seq_len, (batch_size,), device=self.device)

        # Generate new tokens from the position onwards
        proposal_ids = current_ids.clone()

        # Process each batch item separately to handle different positions
        for b in range(batch_size):
            for i in range(pos[b], seq_len):
                logits = self._get_logits(model, proposal_ids[b : b + 1, :i])
                logits = logits / self.proposal_temp
                next_token = self._sample_from_logits(logits)
                proposal_ids[b, i] = next_token.squeeze(-1)

        return proposal_ids

    def _compute_reward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute reward for the given sequence using the reward model.

        Args:
            input_ids: Sequence of token IDs

        Returns:
            torch.Tensor: Reward scores
        """
        with torch.no_grad():
            reward = self.reward_model(input_ids)
        return reward
