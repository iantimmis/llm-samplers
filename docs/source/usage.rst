Usage
=====

This page provides examples of how to use the LLM Samplers library with different types of language models.

Basic Usage
----------

The library provides several sampling methods that can be used with any PyTorch-based language model. Here's a basic example using a Hugging Face model:

.. code-block:: python

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

Using with Custom PyTorch Models
------------------------------

The library works with any PyTorch model that follows a simple interface. Here's an example of using it with a custom PyTorch model:

.. code-block:: python

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

Using with Other PyTorch Models
-----------------------------

You can also use the library with other PyTorch models by creating a simple wrapper that matches the required interface:

.. code-block:: python

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

Model Compatibility
-----------------

The library is designed to work with any PyTorch-based language model that follows these requirements:

1. The model must be callable with input_ids (PyTorch tensor)
2. The model must return an object with a `logits` attribute
3. The model must have a `config` attribute with an `eos_token_id`

This makes it compatible with:
- Hugging Face models
- Custom PyTorch models
- Other PyTorch-based language models (with a simple wrapper)

Available Samplers
----------------

Temperature Scaling
~~~~~~~~~~~~~~~~~

Adjusts the "sharpness" of the probability distribution:

.. code-block:: python

    from llm_samplers import TemperatureSampler

    # More deterministic (picks high-probability tokens)
    sampler = TemperatureSampler(temperature=0.7)

    # More random (flatter distribution)
    sampler = TemperatureSampler(temperature=1.2)

Top-K Sampling
~~~~~~~~~~~~~

Considers only the 'k' most probable tokens:

.. code-block:: python

    from llm_samplers import TopKSampler

    # Consider only top 50 tokens
    sampler = TopKSampler(k=50)

Top-P (Nucleus) Sampling
~~~~~~~~~~~~~~~~~~~~~~~

Selects the smallest set of tokens whose cumulative probability exceeds threshold 'p':

.. code-block:: python

    from llm_samplers import TopPSampler

    # Consider tokens that make up 95% of the probability mass
    sampler = TopPSampler(p=0.95)

Min-P Sampling
~~~~~~~~~~~~~

Dynamically adjusts the sampling pool size based on the probability of the most likely token:

.. code-block:: python

    from llm_samplers import MinPSampler

    # Use min-p sampling with threshold 0.05
    sampler = MinPSampler(min_p=0.05)

Anti-Slop Sampling
~~~~~~~~~~~~~~~~

Down-weights probabilities at word & phrase level:

.. code-block:: python

    from llm_samplers import AntiSlopSampler

    # Initialize with default parameters
    sampler = AntiSlopSampler()

XTC (Exclude Top Choices) Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enhances creativity by nudging the model away from its most predictable choices:

.. code-block:: python

    from llm_samplers import XTCSampler

    # Initialize with default parameters
    sampler = XTCSampler()

Beam Search
~~~~~~~~~~

A breadth-first search algorithm that maintains the top k most promising sequences:

.. code-block:: python

    from llm_samplers import BeamSearchSampler

    # Initialize with beam width 5
    sampler = BeamSearchSampler(beam_width=5)
    
    # Generate with multiple return sequences
    output_ids = sampler.sample(model, input_ids, max_length=100, num_return_sequences=3)

For more detailed information about each sampler's parameters and behavior, see the :doc:`API Reference <api>`. 