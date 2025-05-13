import torch

from llm_samplers.beam import BeamSearchSampler


def test_beam_search_initialization():
    """Test beam search sampler initialization."""
    sampler = BeamSearchSampler(beam_width=3)
    assert sampler.beam_width == 3

    # Test default beam width
    sampler = BeamSearchSampler()
    assert sampler.beam_width == 5


def test_beam_search_sampling(dummy_model, sample_input_ids):
    """Test beam search sampling with a dummy model."""
    sampler = BeamSearchSampler(beam_width=2)
    max_length = 10
    num_return_sequences = 2

    # Test single sequence generation
    output = sampler.sample(
        dummy_model,
        sample_input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
    )

    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == sample_input_ids.shape[0]  # batch size
    assert output.shape[1] == num_return_sequences  # number of sequences
    assert output.shape[2] == max_length  # sequence length


def test_beam_search_batch_processing(dummy_model):
    """Test beam search with batch processing."""
    sampler = BeamSearchSampler(beam_width=2)
    batch_input_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    max_length = 8
    num_return_sequences = 2

    output = sampler.sample(
        dummy_model,
        batch_input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
    )

    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == batch_input_ids.shape[0]  # batch size
    assert output.shape[1] == num_return_sequences  # number of sequences
    assert output.shape[2] == max_length  # sequence length


def test_beam_search_sequence_length(dummy_model, sample_input_ids):
    """Test that beam search respects max_length parameter."""
    sampler = BeamSearchSampler(beam_width=2)
    max_length = 7
    num_return_sequences = 1

    output = sampler.sample(
        dummy_model,
        sample_input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
    )

    assert output.shape[2] == max_length


def test_beam_search_return_sequences(dummy_model, sample_input_ids):
    """Test that beam search returns the correct number of sequences."""
    sampler = BeamSearchSampler(beam_width=3)
    max_length = 10
    num_return_sequences = 2

    output = sampler.sample(
        dummy_model,
        sample_input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
    )

    assert output.shape[1] == num_return_sequences
