import torch

from llm_samplers.base import BaseSampler


class DummyModel:
    def __init__(self, vocab_size=1000):
        self.config = type("Config", (), {"eos_token_id": 0})()
        self.vocab_size = vocab_size

    def __call__(self, input_ids):
        class Output:
            def __init__(self, vocab_size):
                self.logits = torch.randn(1, 1, vocab_size)

        return Output(self.vocab_size)


class TestSampler(BaseSampler):
    """Concrete implementation of BaseSampler for testing."""

    def sample(
        self, model, input_ids, max_length=100, num_return_sequences=1, **kwargs
    ):
        return input_ids


def test_base_sampler_initialization():
    """Test that the base sampler initializes correctly."""
    sampler = TestSampler()
    assert sampler.device in [torch.device("cuda"), torch.device("cpu")]


def test_get_logits():
    """Test that _get_logits returns the correct shape."""
    sampler = TestSampler()
    model = DummyModel(vocab_size=1000)
    input_ids = torch.randint(0, 1000, (1, 10))

    logits = sampler._get_logits(model, input_ids)
    assert logits.shape == (1, 1000)  # batch_size x vocab_size


def test_sample_from_logits():
    """Test that _sample_from_logits returns the correct shape."""
    sampler = TestSampler()
    logits = torch.randn(1, 1000)  # batch_size x vocab_size

    # Test single sample
    samples = sampler._sample_from_logits(logits, num_samples=1)
    assert samples.shape == (1, 1)

    # Test multiple samples
    samples = sampler._sample_from_logits(logits, num_samples=3)
    assert samples.shape == (1, 3)


def test_apply_sampling_default():
    """Test that default _apply_sampling returns unchanged logits."""
    sampler = TestSampler()
    logits = torch.randn(1, 1000)
    filtered_logits = sampler._apply_sampling(logits)
    assert torch.allclose(logits, filtered_logits)
