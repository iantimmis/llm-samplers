import pytest
import torch

from llm_samplers.min_p import MinPSampler


def test_min_p_sampler_initialization():
    """Test Min-P sampler initialization."""
    # Test valid min_p
    sampler = MinPSampler(min_p=0.05)
    assert sampler.min_p == 0.05

    # Test invalid min_p
    with pytest.raises(ValueError):
        MinPSampler(min_p=0)

    with pytest.raises(ValueError):
        MinPSampler(min_p=1)

    with pytest.raises(ValueError):
        MinPSampler(min_p=-0.1)


def test_min_p_sampling():
    """Test Min-P sampling behavior."""
    sampler = MinPSampler(min_p=0.1)
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])

    filtered_logits = sampler._apply_sampling(logits)
    filtered_probs = torch.softmax(filtered_logits, dim=-1)

    # Get the maximum probability
    max_prob = torch.max(filtered_probs, dim=-1)[0]

    # Check that all kept probabilities are above min_p * max_prob
    non_zero_mask = filtered_probs > 0
    kept_probs = filtered_probs[non_zero_mask]
    assert torch.all(kept_probs >= sampler.min_p * max_prob)


def test_min_p_sampling_shape():
    """Test that Min-P sampling maintains tensor shape."""
    sampler = MinPSampler(min_p=0.05)
    logits = torch.randn(2, 1000)  # batch_size x vocab_size
    filtered_logits = sampler._apply_sampling(logits)
    assert filtered_logits.shape == logits.shape


def test_min_p_sampling_batch():
    """Test Min-P sampling with batch processing."""
    sampler = MinPSampler(min_p=0.1)
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]])

    filtered_logits = sampler._apply_sampling(logits)
    filtered_probs = torch.softmax(filtered_logits, dim=-1)

    # Check each sequence in the batch
    for i in range(2):
        max_prob = torch.max(filtered_probs[i], dim=-1)[0]
        non_zero_mask = filtered_probs[i] > 0
        kept_probs = filtered_probs[i][non_zero_mask]
        assert torch.all(kept_probs >= sampler.min_p * max_prob)


def test_min_p_sampling_edge_cases():
    """Test Min-P sampling edge cases."""
    # Test min_p close to 1.0
    sampler = MinPSampler(min_p=0.99)
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    filtered_logits = sampler._apply_sampling(logits)
    filtered_probs = torch.softmax(filtered_logits, dim=-1)

    # Only the highest probability token should be kept
    non_zero_mask = filtered_probs > 0
    assert non_zero_mask.sum() == 1

    # Test min_p close to 0.0
    sampler = MinPSampler(min_p=0.01)
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    filtered_logits = sampler._apply_sampling(logits)
    filtered_probs = torch.softmax(filtered_logits, dim=-1)

    # Most tokens should be kept
    non_zero_mask = filtered_probs > 0
    assert non_zero_mask.sum() > 1
