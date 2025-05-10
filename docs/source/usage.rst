Usage
=====

Basic Usage
----------

Here's a basic example of how to use LLM Samplers with a Hugging Face model:

.. code-block:: python

    from llm_samplers import TemperatureSampler, TopKSampler, TopPSampler
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
    print(generated_text)

Using Different Samplers
-----------------------

LLM Samplers provides several sampling methods, which can be used interchangeably:

.. code-block:: python

    # Temperature sampling
    temp_sampler = TemperatureSampler(temperature=0.8)
    
    # Top-K sampling
    topk_sampler = TopKSampler(k=40)
    
    # Top-P (Nucleus) sampling
    topp_sampler = TopPSampler(p=0.95)
    
    # Min-P sampling
    minp_sampler = MinPSampler(min_p=0.05)
    
    # Anti-Slop sampling
    anti_slop_sampler = AntiSlopSampler()
    
    # XTC (Exclude Top Choices) sampling
    xtc_sampler = XTCSampler(exclude_percent=0.2)

Advanced Usage
------------

For more advanced use cases, you can combine multiple sampling techniques:

.. code-block:: python

    from llm_samplers import TopPSampler, TemperatureSampler
    
    # First apply temperature sampling
    logits = model(input_ids).logits[:, -1, :]
    temp_sampler = TemperatureSampler(temperature=0.8)
    modified_logits = temp_sampler.adjust_logits(logits)
    
    # Then apply Top-P sampling on the temperature-adjusted logits
    topp_sampler = TopPSampler(p=0.95)
    token_id = topp_sampler.sample_token(modified_logits)
    
    # Add the sampled token to input_ids
    input_ids = torch.cat([input_ids, token_id.unsqueeze(0)], dim=1) 