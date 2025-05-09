import pytest
import torch

from llm_samplers.temperature import TemperatureSampler


def test_temperature_sampler_initialization():
    """Test temperature sampler initialization."""
    # Test valid temperature
    sampler = TemperatureSampler(temperature=0.7)
    assert sampler.temperature == 0.7

    # Test invalid temperature
    with pytest.raises(ValueError):
        TemperatureSampler(temperature=0)

    with pytest.raises(ValueError):
        TemperatureSampler(temperature=-1)


def test_temperature_sampling():
    """Test temperature sampling behavior."""
    logits = torch.tensor([[1.0, 2.0, 3.0]])

    # Test that higher temperature makes distribution more uniform
    high_temp_sampler = TemperatureSampler(temperature=2.0)
    high_temp_probs = torch.softmax(high_temp_sampler._apply_sampling(logits), dim=-1)

    # Test that lower temperature makes distribution more peaked
    low_temp_sampler = TemperatureSampler(temperature=0.5)
    low_temp_probs = torch.softmax(low_temp_sampler._apply_sampling(logits), dim=-1)

    # Higher temperature should make probabilities more similar
    assert high_temp_probs.std() < low_temp_probs.std()


def test_temperature_sampling_shape():
    """Test that temperature sampling maintains tensor shape."""
    sampler = TemperatureSampler(temperature=0.7)
    logits = torch.randn(2, 1000)  # batch_size x vocab_size
    filtered_logits = sampler._apply_sampling(logits)
    assert filtered_logits.shape == logits.shape


def test_temperature_sampling_values():
    """Test that temperature scaling is applied correctly."""
    sampler = TemperatureSampler(temperature=0.5)
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    filtered_logits = sampler._apply_sampling(logits)

    # Check that values are scaled by temperature
    expected = logits / 0.5
    assert torch.allclose(filtered_logits, expected)
