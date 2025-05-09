import pytest
import torch

class DummyModel:
    """A dummy model for testing samplers."""
    def __init__(self, vocab_size=1000):
        self.config = type('Config', (), {'eos_token_id': 0})()
        self.vocab_size = vocab_size
    
    def __call__(self, input_ids):
        class Output:
            def __init__(self, vocab_size):
                self.logits = torch.randn(1, 1, vocab_size)
        return Output(self.vocab_size)

@pytest.fixture
def dummy_model():
    """Fixture providing a dummy model for testing."""
    return DummyModel()

@pytest.fixture
def sample_logits():
    """Fixture providing sample logits for testing."""
    return torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])

@pytest.fixture
def batch_logits():
    """Fixture providing batch logits for testing."""
    return torch.tensor([
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1]
    ])

@pytest.fixture
def sample_input_ids():
    """Fixture providing sample input IDs for testing."""
    return torch.tensor([[0, 1, 2, 3, 4]]) 