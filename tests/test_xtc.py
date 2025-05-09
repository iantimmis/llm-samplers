import pytest
import torch

from llm_samplers.xtc import XTCSampler


def test_xtc_sampler_initialization():
    """Test XTC sampler initialization."""
    # Test valid parameters
    sampler = XTCSampler(top_k=5, exclusion_threshold=0.3, min_probability=0.1)
    assert sampler.top_k == 5
    assert sampler.exclusion_threshold == 0.3
    assert sampler.min_probability == 0.1

    # Test invalid top_k
    with pytest.raises(ValueError):
        XTCSampler(top_k=0)

    with pytest.raises(ValueError):
        XTCSampler(top_k=-1)

    # Test invalid exclusion_threshold
    with pytest.raises(ValueError):
        XTCSampler(exclusion_threshold=0)

    with pytest.raises(ValueError):
        XTCSampler(exclusion_threshold=1)

    # Test invalid min_probability
    with pytest.raises(ValueError):
        XTCSampler(min_probability=0)

    with pytest.raises(ValueError):
        XTCSampler(min_probability=1)


def test_xtc_sampling():
    """Test XTC sampling behavior."""
    sampler = XTCSampler(top_k=2, exclusion_threshold=0.3, min_probability=0.1)
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])

    filtered_logits = sampler._apply_sampling(logits)
    filtered_probs = torch.softmax(filtered_logits, dim=-1)

    # Get the top-k probabilities
    top_k_probs, _ = torch.topk(filtered_probs, sampler.top_k, dim=-1)

    # Check that excluded tokens are in top-k and above thresholds
    non_inf_mask = filtered_logits != float("-inf")
    kept_probs = filtered_probs[non_inf_mask]
    assert torch.all(kept_probs <= sampler.exclusion_threshold) or torch.all(
        kept_probs <= sampler.min_probability
    )


def test_xtc_sampling_shape():
    """Test that XTC sampling maintains tensor shape."""
    sampler = XTCSampler(top_k=5, exclusion_threshold=0.3, min_probability=0.1)
    logits = torch.randn(2, 1000)  # batch_size x vocab_size
    filtered_logits = sampler._apply_sampling(logits)
    assert filtered_logits.shape == logits.shape


def test_xtc_sampling_batch():
    """Test XTC sampling with batch processing."""
    sampler = XTCSampler(top_k=2, exclusion_threshold=0.3, min_probability=0.1)
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]])

    filtered_logits = sampler._apply_sampling(logits)
    filtered_probs = torch.softmax(filtered_logits, dim=-1)

    # Check each sequence in the batch
    for i in range(2):
        non_inf_mask = filtered_logits[i] != float("-inf")
        kept_probs = filtered_probs[i][non_inf_mask]
        assert torch.all(kept_probs <= sampler.exclusion_threshold) or torch.all(
            kept_probs <= sampler.min_probability
        )


def test_xtc_sampling_edge_cases():
    """Test XTC sampling edge cases."""
    # Test with high exclusion threshold
    sampler = XTCSampler(top_k=2, exclusion_threshold=0.9, min_probability=0.1)
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    filtered_logits = sampler._apply_sampling(logits)
    filtered_probs = torch.softmax(filtered_logits, dim=-1)

    # Most tokens should be kept
    non_zero_mask = filtered_probs > 0
    assert non_zero_mask.sum() > 1

    # Test with low exclusion threshold
    sampler = XTCSampler(top_k=2, exclusion_threshold=0.1, min_probability=0.1)
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    filtered_logits = sampler._apply_sampling(logits)
    filtered_probs = torch.softmax(filtered_logits, dim=-1)

    # Fewer tokens should be kept
    non_zero_mask = filtered_probs > 0
    assert non_zero_mask.sum() <= 3
