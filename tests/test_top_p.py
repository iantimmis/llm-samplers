import pytest
import torch
from src.samplers.top_p import TopPSampler

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
    
    # Convert to probabilities to check cumulative sum
    probs = torch.softmax(logits, dim=-1)
    filtered_probs = torch.softmax(filtered_logits, dim=-1)
    
    # Check that cumulative probability of kept tokens is <= p
    sorted_probs, _ = torch.sort(filtered_probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    assert torch.all(cumulative_probs <= sampler.p)

def test_top_p_sampling_shape():
    """Test that Top-P sampling maintains tensor shape."""
    sampler = TopPSampler(p=0.9)
    logits = torch.randn(2, 1000)  # batch_size x vocab_size
    filtered_logits = sampler._apply_sampling(logits)
    assert filtered_logits.shape == logits.shape

def test_top_p_sampling_batch():
    """Test Top-P sampling with batch processing."""
    sampler = TopPSampler(p=0.6)
    logits = torch.tensor([
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1]
    ])
    
    filtered_logits = sampler._apply_sampling(logits)
    filtered_probs = torch.softmax(filtered_logits, dim=-1)
    
    # Check each sequence in the batch
    for i in range(2):
        sorted_probs, _ = torch.sort(filtered_probs[i], descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        assert torch.all(cumulative_probs <= sampler.p)

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