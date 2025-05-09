import pytest
import torch

from llm_samplers.anti_slop import AntiSlopSampler


def test_anti_slop_sampler_initialization():
    """Test Anti-Slop sampler initialization."""
    # Test valid parameters
    sampler = AntiSlopSampler(
        disallowed_tokens={1, 2, 3},
        disallowed_phrases=[[1, 2], [3, 4]],
        penalty=0.5,
        max_retries=3,
    )
    assert sampler.disallowed_tokens == {1, 2, 3}
    assert sampler.disallowed_phrases == [[1, 2], [3, 4]]
    assert sampler.penalty == 0.5
    assert sampler.max_retries == 3

    # Test invalid penalty
    with pytest.raises(ValueError):
        AntiSlopSampler(penalty=-0.1)

    with pytest.raises(ValueError):
        AntiSlopSampler(penalty=1.1)

    # Test invalid max_retries
    with pytest.raises(ValueError):
        AntiSlopSampler(max_retries=0)


def test_anti_slop_sampling():
    """Test Anti-Slop sampling behavior."""
    sampler = AntiSlopSampler(disallowed_tokens={1, 2}, penalty=0.5)
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

    filtered_logits = sampler._apply_sampling(logits)

    # Check that disallowed tokens are penalized
    assert filtered_logits[0, 1] == logits[0, 1] * 0.5
    assert filtered_logits[0, 2] == logits[0, 2] * 0.5

    # Check that other tokens are unchanged
    assert filtered_logits[0, 0] == logits[0, 0]
    assert filtered_logits[0, 3] == logits[0, 3]
    assert filtered_logits[0, 4] == logits[0, 4]


def test_anti_slop_sampling_shape():
    """Test that Anti-Slop sampling maintains tensor shape."""
    sampler = AntiSlopSampler(disallowed_tokens={1, 2})
    logits = torch.randn(2, 1000)  # batch_size x vocab_size
    filtered_logits = sampler._apply_sampling(logits)
    assert filtered_logits.shape == logits.shape


def test_anti_slop_phrase_checking():
    """Test Anti-Slop phrase checking functionality."""
    sampler = AntiSlopSampler(disallowed_phrases=[[1, 2], [3, 4]], max_retries=2)

    # Test sequence with disallowed phrase
    sequence = torch.tensor([[0, 1, 2, 5, 6]])
    assert sampler._check_phrases(sequence)

    # Test sequence without disallowed phrase
    sequence = torch.tensor([[0, 1, 3, 5, 6]])
    assert not sampler._check_phrases(sequence)


def test_anti_slop_sampling_batch():
    """Test Anti-Slop sampling with batch processing."""
    sampler = AntiSlopSampler(disallowed_tokens={1, 2}, penalty=0.5)
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]])

    filtered_logits = sampler._apply_sampling(logits)

    # Check each sequence in the batch
    for i in range(2):
        assert filtered_logits[i, 1] == logits[i, 1] * 0.5
        assert filtered_logits[i, 2] == logits[i, 2] * 0.5


def test_anti_slop_retry_behavior():
    """Test Anti-Slop retry behavior."""

    class DummyModel:
        def __init__(self):
            self.config = type("Config", (), {"eos_token_id": 0})()

        def __call__(self, input_ids):
            class Output:
                def __init__(self):
                    # Add batch and sequence dimensions to match model output format
                    self.logits = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]])

            return Output()

    sampler = AntiSlopSampler(disallowed_phrases=[[1, 2]], max_retries=2)
    model = DummyModel()
    input_ids = torch.tensor([[0, 1, 2]])

    # The sampler should retry when it encounters a disallowed phrase
    output_ids = sampler.sample(model, input_ids, max_length=5)
    assert output_ids.shape[1] <= 5  # Should not exceed max_length
