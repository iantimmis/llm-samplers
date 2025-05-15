Examples
========

Here are some practical examples showing how to use LLM Samplers in various scenarios.

Basic Text Generation
-------------------

.. code-block:: python

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llm_samplers import TemperatureSampler

    # Load model and tokenizer
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize sampler
    sampler = TemperatureSampler(temperature=0.7)

    # Setup input
    input_text = "Once upon a time"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate text
    max_length = 50
    output_ids = sampler.sample(model, input_ids, max_new_tokens=max_length)
    
    # Decode and print result
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(generated_text)

Comparing Different Samplers
--------------------------

.. code-block:: python

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llm_samplers import (
        TemperatureSampler, 
        TopKSampler, 
        TopPSampler, 
        MinPSampler
    )

    # Load model and tokenizer
    model_name = "gpt2-medium"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize samplers
    samplers = {
        "Temperature (0.7)": TemperatureSampler(temperature=0.7),
        "Top-K (40)": TopKSampler(k=40),
        "Top-P (0.95)": TopPSampler(p=0.95),
        "Min-P (0.05)": MinPSampler(min_p=0.05),
    }

    # Setup input
    input_text = "The future of artificial intelligence is"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate text with each sampler
    max_length = 100
    results = {}

    for name, sampler in samplers.items():
        output_ids = sampler.sample(model, input_ids, max_new_tokens=max_length)
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        results[name] = generated_text
        
    # Print results
    for name, text in results.items():
        print(f"\n=== {name} ===")
        print(text)

Customizing Generation Parameters
------------------------------

.. code-block:: python

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llm_samplers import TopPSampler

    # Load model and tokenizer
    model_name = "gpt2-large"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize sampler with custom parameters
    sampler = TopPSampler(
        p=0.92,
        temperature=0.8,  # You can combine Top-P with temperature
        repetition_penalty=1.2  # Penalize repeated tokens
    )

    # Setup input
    input_text = "Write a short poem about"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate with custom parameters
    output_ids = sampler.sample(
        model,
        input_ids,
        max_new_tokens=150,
        min_new_tokens=50,  # Force minimum generation length
        no_repeat_ngram_size=3  # Prevent repeating 3-grams
    )
    
    # Decode and print result
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(generated_text)

Using Anti-Slop for Higher Quality
--------------------------------

.. code-block:: python

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llm_samplers import AntiSlopSampler

    # Load model and tokenizer
    model_name = "gpt2-xl"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize Anti-Slop sampler
    sampler = AntiSlopSampler(
        word_level=True,  # Enable word-level backtracking
        backtrack_threshold=0.4,  # Threshold for backtracking
        max_retries=3  # Maximum number of retries
    )

    # Setup input
    input_text = "Explain the concept of quantum computing in simple terms:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate text
    output_ids = sampler.sample(model, input_ids, max_new_tokens=200)
    
    # Decode and print result
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(generated_text)

Using Beam Search for Deterministic Outputs
-----------------------------------------

.. code-block:: python

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llm_samplers import BeamSearchSampler

    # Load model and tokenizer
    model_name = "gpt2-medium"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize Beam Search sampler
    sampler = BeamSearchSampler(beam_width=5)

    # Setup input
    input_text = "Summarize the key benefits of renewable energy:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate multiple sequences
    output_ids = sampler.sample(
        model,
        input_ids,
        max_length=150,
        num_return_sequences=3  # Return 3 different outputs
    )
    
    # Decode and print results
    for i, ids in enumerate(output_ids[0]):
        generated_text = tokenizer.decode(ids, skip_special_tokens=True)
        print(f"\n=== Sequence {i+1} ===")
        print(generated_text) 