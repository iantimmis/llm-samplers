import pytest
import torch

from llm_samplers.top_p import TopPSampler


def test_top_p_sampler_initialization():
    """Test Top-P sampler initialization."""
    # Test valid p
    sampler = TopPSampler(p=0.9)
    assert sampler.p == 0.9

    # Test invalid p
    with pytest.raises(ValueError):
        TopPSampler(p=0)

    with pytest.raises(ValueError):
        TopPSampler(p=1.1)

    with pytest.raises(ValueError):
        TopPSampler(p=-0.1)


def test_top_p_sampling():
    """Test Top-P sampling behavior."""
    sampler = TopPSampler(p=0.6)
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])

    filtered_logits = sampler._apply_sampling(logits)

    # Convert to probabilities
    filtered_probs = torch.softmax(filtered_logits, dim=-1)

    # Get non-zero probabilities (not -inf in logits)
    non_zero_probs = filtered_probs[filtered_probs > 0]

    # Calculate cumulative sum of the kept probabilities
    cumsum = torch.cumsum(non_zero_probs, dim=0)

    # The last cumulative probability should be close to 1
    assert torch.isclose(cumsum[-1], torch.tensor(1.0), rtol=1e-3)

    # The number of non-zero probabilities should be minimal while sum is > p
    assert cumsum[0] <= sampler.p
    if len(cumsum) > 1:
        assert cumsum[1] > sampler.p


def test_top_p_sampling_shape():
    """Test that Top-P sampling maintains tensor shape."""
    sampler = TopPSampler(p=0.9)
    logits = torch.randn(2, 1000)  # batch_size x vocab_size
    filtered_logits = sampler._apply_sampling(logits)
    assert filtered_logits.shape == logits.shape


def test_top_p_sampling_batch():
    """Test Top-P sampling with batch processing."""
    sampler = TopPSampler(p=0.6)
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]])

    filtered_logits = sampler._apply_sampling(logits)
    filtered_probs = torch.softmax(filtered_logits, dim=-1)

    # Check each sequence in the batch
    for i in range(2):
        # Get non-zero probabilities
        non_zero_probs = filtered_probs[i][filtered_probs[i] > 0]

        # Calculate cumulative sum
        cumsum = torch.cumsum(non_zero_probs, dim=0)

        # The last cumulative probability should be close to 1
        assert torch.isclose(cumsum[-1], torch.tensor(1.0), rtol=1e-3)

        # The number of non-zero probabilities should be minimal while sum is > p
        assert cumsum[0] <= sampler.p
        if len(cumsum) > 1:
            assert cumsum[1] > sampler.p


def test_top_p_sampling_edge_cases():
    """Test Top-P sampling edge cases."""
    # Test p = 1.0 (keep all tokens)
    sampler = TopPSampler(p=1.0)
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    filtered_logits = sampler._apply_sampling(logits)
    assert torch.allclose(filtered_logits, logits)

    # Test very small p
    sampler = TopPSampler(p=0.1)
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    filtered_logits = sampler._apply_sampling(logits)
    filtered_probs = torch.softmax(filtered_logits, dim=-1)

    # Check that only the highest probability token is kept
    non_zero_mask = filtered_probs > 0
    assert non_zero_mask.sum() == 1
