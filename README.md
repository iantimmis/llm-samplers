# LLM Samplers

A Python library for advanced LLM sampling techniques, providing a collection of sophisticated sampling methods for language models.

## Features

- Temperature Scaling
- Top-K Sampling
- Top-P (Nucleus) Sampling
- Min-P Sampling
- Anti-Slop Sampling
- XTC (Exclude Top Choices) Sampling

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

This project uses Ruff for linting and code formatting. To check your code:

```bash
# Run Ruff linter
ruff check .

# Run Ruff formatter
ruff format .
```

The project is configured with a GitHub Action that automatically runs Ruff on all pull requests.

## Usage

```python
from llm_samplers import TemperatureSampler, TopKSampler, TopPSampler, MinPSampler, AntiSlopSampler, XTCSampler

# Initialize a sampler
sampler = TemperatureSampler(temperature=0.7)

# Use with a HuggingFace model
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Generate text with the sampler
input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output_ids = sampler.sample(model, input_ids)
generated_text = tokenizer.decode(output_ids[0])
```

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

## License

MIT License
