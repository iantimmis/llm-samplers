import pytest
import torch
from torch import nn

from llm_samplers.qalign import QAlignSampler


class DummyRewardModel(nn.Module):
    """A dummy reward model for testing QAlign."""

    def __init__(self, bias=0.0):
        super().__init__()
        self.bias = bias

    def forward(self, input_ids):
        # Return a simple reward based on the sum of input_ids
        return torch.sum(input_ids.float(), dim=1) + self.bias


@pytest.fixture
def dummy_reward_model():
    """Fixture providing a dummy reward model for testing."""
    return DummyRewardModel()


def test_qalign_sampler_initialization():
    """Test QAlign sampler initialization."""
    reward_model = DummyRewardModel()

    # Test valid initialization
    sampler = QAlignSampler(
        reward_model=reward_model,
        num_steps=10,
        temperature=1.0,
        beta=1.0,
        proposal_temp=1.0,
    )
    assert sampler.num_steps == 10
    assert sampler.temperature == 1.0
    assert sampler.beta == 1.0
    assert sampler.proposal_temp == 1.0

    # Test invalid temperature
    with pytest.raises(ValueError):
        QAlignSampler(reward_model=reward_model, temperature=0)

    # Test invalid proposal temperature
    with pytest.raises(ValueError):
        QAlignSampler(reward_model=reward_model, proposal_temp=0)


def test_qalign_proposal_generation(dummy_model, dummy_reward_model):
    """Test proposal generation in QAlign."""
    sampler = QAlignSampler(
        reward_model=dummy_reward_model,
        num_steps=1,
        temperature=1.0,
        beta=1.0,
        proposal_temp=1.0,
    )

    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    proposal_ids = sampler._generate_proposal(dummy_model, input_ids)

    # Check that proposal has same shape as input
    assert proposal_ids.shape == input_ids.shape
    # Check that proposal is different from input (since we're resampling)
    assert not torch.allclose(proposal_ids, input_ids)


def test_qalign_reward_computation(dummy_reward_model):
    """Test reward computation in QAlign."""
    sampler = QAlignSampler(
        reward_model=dummy_reward_model,
        num_steps=1,
        temperature=1.0,
        beta=1.0,
        proposal_temp=1.0,
    )

    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    reward = sampler._compute_reward(input_ids)

    # Check reward shape (should be batch_size)
    assert reward.shape == (1,)
    # Check reward value (sum of input_ids)
    assert reward.item() == 15.0


def test_qalign_sampling(dummy_model, dummy_reward_model):
    """Test full QAlign sampling process."""
    sampler = QAlignSampler(
        reward_model=dummy_reward_model,
        num_steps=5,  # Use a small number of steps for testing
        temperature=1.0,
        beta=1.0,
        proposal_temp=1.0,
    )

    input_ids = torch.tensor([[1, 2, 3]])
    output_ids = sampler.sample(
        model=dummy_model,
        input_ids=input_ids,
        max_length=5,
    )

    # Check output shape
    assert output_ids.shape[0] == input_ids.shape[0]  # batch size
    assert output_ids.shape[1] >= input_ids.shape[1]  # sequence length
    assert output_ids.shape[1] <= 5  # max_length


def test_qalign_batch_processing(dummy_model, dummy_reward_model):
    """Test QAlign with batch inputs."""
    sampler = QAlignSampler(
        reward_model=dummy_reward_model,
        num_steps=5,
        temperature=1.0,
        beta=1.0,
        proposal_temp=1.0,
    )

    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])  # batch size of 2
    output_ids = sampler.sample(
        model=dummy_model,
        input_ids=input_ids,
        max_length=5,
    )

    # Check batch processing
    assert output_ids.shape[0] == 2  # batch size preserved
    assert output_ids.shape[1] >= input_ids.shape[1]  # sequence length
    assert output_ids.shape[1] <= 5  # max_length


def test_qalign_reward_influence(dummy_model):
    """Test that higher beta values lead to more reward-focused sampling."""
    # Create two reward models with different biases
    high_reward_model = DummyRewardModel(bias=10.0)
    low_reward_model = DummyRewardModel(bias=0.0)

    # Create samplers with different beta values
    high_beta_sampler = QAlignSampler(
        reward_model=high_reward_model,
        num_steps=5,
        temperature=1.0,
        beta=2.0,  # Higher beta
        proposal_temp=1.0,
    )

    low_beta_sampler = QAlignSampler(
        reward_model=low_reward_model,
        num_steps=5,
        temperature=1.0,
        beta=0.5,  # Lower beta
        proposal_temp=1.0,
    )

    input_ids = torch.tensor([[1, 2, 3]])

    # Sample from both samplers
    high_beta_output = high_beta_sampler.sample(
        model=dummy_model,
        input_ids=input_ids,
        max_length=5,
    )

    low_beta_output = low_beta_sampler.sample(
        model=dummy_model,
        input_ids=input_ids,
        max_length=5,
    )

    # Check that outputs are different due to different reward models and beta values
    assert not torch.allclose(high_beta_output, low_beta_output)
