# LLM Samplers

A Python library for advanced LLM sampling techniques, providing a collection of sophisticated sampling methods for language models. This library is model-agnostic and works with any PyTorch-based language model that follows a simple interface.

[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue)](https://llm-samplers.readthedocs.io/)

## Features

- Temperature Scaling
- Top-K Sampling
- Top-P (Nucleus) Sampling
- Min-P Sampling
- Anti-Slop Sampling
- XTC (Exclude Top Choices) Sampling
- QAlign (MCMC Test-Time Alignment) Sampling
- Model-agnostic: Works with any PyTorch-based language model
- Compatible with Hugging Face models and custom implementations

## Installation

### From PyPI

```bash
pip install llm-samplers
```

### From Source

1. Clone the repository:

```bash
git clone https://github.com/iantimmis/llm-samplers.git
cd samplers
```

2. Create and activate a virtual environment (recommended):

```bash
# Using venv
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows

# Using uv (recommended)
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

3. Install the package in development mode:

```bash
# Using pip
pip install -e ".[dev]"  # Includes development dependencies

# Using uv (recommended)
uv pip install -e .  # uv installs dev dependencies by default
```

## Documentation

For detailed documentation, visit [llm-samplers.readthedocs.io](https://llm-samplers.readthedocs.io/).

## Development

### Running Tests

The test suite uses pytest. To run the tests:

```bash
# Run all tests
python -m pytest tests/

# Run tests with verbose output
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_temperature.py
```

### Code Quality

This project uses Ruff for linting and formatting. To check your code:

```bash
# Run Ruff linter
ruff check .

# Format your code
ruff format .
```

The project is configured with a GitHub Action that automatically runs Ruff on all pull requests.

## Usage

### With Hugging Face Models

```python
from llm_samplers import TemperatureSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Initialize a sampler
sampler = TemperatureSampler(temperature=0.7)

# Generate text with the sampler
input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output_ids = sampler.sample(model, input_ids)
generated_text = tokenizer.decode(output_ids[0])
```

### With Custom PyTorch Models

The library works with any PyTorch model that follows this interface:

```python
import torch
from llm_samplers import TemperatureSampler

class CustomLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size=1000):
        super().__init__()
        self.config = type("Config", (), {"eos_token_id": 0})()
        self.vocab_size = vocab_size
        # Your model architecture here
        self.embedding = torch.nn.Embedding(vocab_size, 512)
        self.transformer = torch.nn.TransformerEncoder(...)
        self.output = torch.nn.Linear(512, vocab_size)

    def forward(self, input_ids):
        # Your model's forward pass here
        x = self.embedding(input_ids)
        x = self.transformer(x)
        logits = self.output(x)
        return type("Output", (), {"logits": logits})()

# Initialize model and sampler
model = CustomLanguageModel()
sampler = TemperatureSampler(temperature=0.7)

# Generate text
input_ids = torch.tensor([[1, 2, 3]])  # Your input token IDs
output_ids = sampler.sample(model, input_ids)
```

### With Other PyTorch Models

The library can also work with other PyTorch models by wrapping them to match the required interface:

```python
import torch
from llm_samplers import TopPSampler

class ModelWrapper:
    def __init__(self, base_model, tokenizer):
        self.model = base_model
        self.tokenizer = tokenizer
        self.config = type("Config", (), {"eos_token_id": tokenizer.eos_token_id})()

    def __call__(self, input_ids):
        # Adapt your model's output to match the required interface
        outputs = self.model(input_ids)
        return type("Output", (), {"logits": outputs.logits})()

# Initialize your model and wrapper
base_model = YourPyTorchModel()
tokenizer = YourTokenizer()
model = ModelWrapper(base_model, tokenizer)

# Use with samplers
sampler = TopPSampler(p=0.95)
input_ids = tokenizer.encode("Your input text", return_tensors="pt")
output_ids = sampler.sample(model, input_ids)
```

For more examples and detailed usage instructions, see the [documentation](https://llm-samplers.readthedocs.io/).

## Available Samplers

### Temperature Scaling

Adjusts the "sharpness" of the probability distribution:

- Low temperature (<1.0): More deterministic, picks high-probability tokens
- High temperature (>1.0): More random, flatter distribution

### Top-K Sampling

Considers only the 'k' most probable tokens, filtering out unlikely ones.

### Top-P (Nucleus) Sampling

Selects the smallest set of tokens whose cumulative probability exceeds threshold 'p'.

### Min-P Sampling

Dynamically adjusts the sampling pool size based on the probability of the most likely token.

### Anti-Slop

Down-weights probabilities at word & phrase level, using backtracking to retry with adjusted probabilities.

### XTC (Exclude Top Choices)

Enhances creativity by nudging the model away from its most predictable choices.

### QAlign

Uses Markov Chain Monte Carlo (MCMC) to align model outputs with a reward model at test time without fine-tuning.

Based on the paper: ["Sample, Don't Search: Rethinking Test-Time Alignment for Language Models"](https://arxiv.org/abs/2504.03790) (Faria et al., 2024)

For detailed information about each sampler, visit the [documentation](https://llm-samplers.readthedocs.io/).

## Model Compatibility

The library is designed to work with any PyTorch-based language model that follows a simple interface:

1. The model must be callable with input_ids (PyTorch tensor)
2. The model must return an object with a `logits` attribute
3. The model must have a `config` attribute with an `eos_token_id`

This makes it compatible with:

- Hugging Face models
- Custom PyTorch models
- Other PyTorch-based language models (with a simple wrapper)

## License

MIT License
