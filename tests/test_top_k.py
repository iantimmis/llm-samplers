import pytest
import torch

from llm_samplers.top_k import TopKSampler


def test_top_k_sampler_initialization():
    """Test Top-K sampler initialization."""
    # Test valid k
    sampler = TopKSampler(k=50)
    assert sampler.k == 50

    # Test invalid k
    with pytest.raises(ValueError):
        TopKSampler(k=0)

    with pytest.raises(ValueError):
        TopKSampler(k=-1)


def test_top_k_sampling():
    """Test Top-K sampling behavior."""
    sampler = TopKSampler(k=2)
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

    filtered_logits = sampler._apply_sampling(logits)

    # Check that only top-k values are kept
    non_inf_mask = filtered_logits != float("-inf")
    assert non_inf_mask.sum() == 2  # Only 2 values should be kept

    # Check that the kept values are the highest ones
    kept_values = filtered_logits[non_inf_mask]
    assert torch.allclose(kept_values, torch.tensor([4.0, 5.0]))


def test_top_k_sampling_shape():
    """Test that Top-K sampling maintains tensor shape."""
    sampler = TopKSampler(k=50)
    logits = torch.randn(2, 1000)  # batch_size x vocab_size
    filtered_logits = sampler._apply_sampling(logits)
    assert filtered_logits.shape == logits.shape


def test_top_k_sampling_batch():
    """Test Top-K sampling with batch processing."""
    sampler = TopKSampler(k=2)
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]])

    filtered_logits = sampler._apply_sampling(logits)

    # Check each sequence in the batch
    for i in range(2):
        non_inf_mask = filtered_logits[i] != float("-inf")
        assert non_inf_mask.sum() == 2  # Only 2 values should be kept per sequence


def test_top_k_sampling_edge_cases():
    """Test Top-K sampling edge cases."""
    # Test k equal to vocabulary size
    sampler = TopKSampler(k=5)
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    filtered_logits = sampler._apply_sampling(logits)
    assert torch.allclose(filtered_logits, logits)  # All values should be kept

    # Test k larger than vocabulary size
    sampler = TopKSampler(k=10)
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    filtered_logits = sampler._apply_sampling(logits)
    assert torch.allclose(filtered_logits, logits)  # All values should be kept
