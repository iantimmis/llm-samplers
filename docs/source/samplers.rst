Sampling Methods
===============

This library provides various sampling methods for language models. Each method has its own characteristics and is suitable for different use cases.

Temperature Sampling
------------------

Temperature sampling adjusts the "sharpness" of the probability distribution:

- Low temperature (< 1.0): More deterministic outputs, focusing on high-probability tokens
- High temperature (> 1.0): More random outputs, flattens the distribution

Temperature sampling works by dividing the logits by the temperature value before applying the softmax function:

.. math::

   p(x_i) = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}

where :math:`z_i` are the logits and :math:`T` is the temperature.

**When to use:**
- For controlling the randomness/creativity of outputs
- As a base sampling method that can be combined with others

Top-K Sampling
------------

Top-K sampling restricts the sampling to only the K most likely tokens at each step, filtering out unlikely tokens.

The algorithm:
1. Sort tokens by their probability
2. Keep only the top K tokens
3. Renormalize the probabilities of these K tokens
4. Sample from this reduced set

**When to use:**
- When you want to eliminate low-probability tokens
- For more focused and coherent text generation
- When you need a simple method to reduce randomness

Top-P (Nucleus) Sampling
---------------------

Top-P sampling (also known as nucleus sampling) keeps the smallest set of tokens whose cumulative probability exceeds a threshold p:

1. Sort tokens by decreasing probability
2. Keep adding tokens to the set until their cumulative probability exceeds p
3. Renormalize the probabilities of tokens in this set
4. Sample from this dynamic set

**When to use:**
- For a more adaptive approach than Top-K
- To maintain diversity while removing very unlikely tokens
- In scenarios where distribution varies significantly between steps

Min-P Sampling
-----------

Min-P sampling keeps all tokens whose probability is at least p * (probability of the most likely token):

1. Find the probability of the most likely token (p_max)
2. Keep all tokens whose probability is at least min_p * p_max
3. Renormalize probabilities of tokens in this set
4. Sample from this set

**When to use:**
- When the absolute probability matters more than relative ranking
- For maintaining probability mass among relatively likely candidates
- As an alternative to Top-P when you want a more relative threshold

Anti-Slop Sampling
---------------

Anti-Slop is a technique designed to improve the quality of generated text by detecting and preventing "slop" (low-quality, repetitive, or nonsensical content):

1. Apply backtracking at the word or phrase level when detecting low-quality outputs
2. Down-weight probabilities for problematic sequences
3. Retry with adjusted probabilities

**When to use:**
- For higher-quality text generation
- To reduce repetition and nonsensical outputs
- In applications where output quality is critical

XTC (Exclude Top Choices) Sampling
-------------------------------

XTC sampling nudges the model away from its most predictable choices by excluding a percentage of the top-weighted tokens:

1. Sort tokens by decreasing probability
2. Exclude the top N% of tokens (by probability mass)
3. Renormalize the remaining tokens
4. Sample from this set

**When to use:**
- To enhance creativity and diversity
- When standard outputs are too predictable
- For applications requiring novel or surprising content 

QAlign Sampling
------------

QAlign is a test-time alignment method that uses Markov Chain Monte Carlo (MCMC) to improve model outputs based on a reward model.

This method is based on the research paper:

**"Sample, Don't Search: Rethinking Test-Time Alignment for Language Models"**
  Gonçalo Faria, Noah A. Smith (2024)
  Paper: https://arxiv.org/abs/2504.03790

The algorithm works as follows:

1. Generate an initial sequence using the base language model
2. Perform MCMC steps with Metropolis-Hastings acceptance:
   a. Generate a proposal by resampling a portion of the sequence
   b. Compute rewards for current and proposed sequences
   c. Accept proposal with probability min(1, exp(β * (proposal_reward - current_reward)))
3. Return the final sequence after MCMC iterations

Unlike other test-time optimization methods that search for a single optimal output, QAlign converges to sampling from the optimal aligned distribution for each prompt as compute scales. This prevents over-optimization of imperfect reward models.

**When to use:**
- For aligning model outputs with specific objectives without fine-tuning
- When you have a reward model that can score text quality
- To improve model performance on specific tasks at inference time
- As an alternative to computationally expensive fine-tuning approaches 

Beam Search
----------

Beam search is a breadth-first search algorithm that maintains the top k most promising sequences at each step:

1. Start with the initial sequence
2. At each step:
   a. Generate all possible next tokens for each sequence
   b. Score each new sequence using log probabilities
   c. Keep only the top k sequences
3. Return the best sequences after reaching max_length

The algorithm uses a beam width parameter to control how many sequences are maintained at each step. A larger beam width explores more possibilities but requires more computation.

**When to use:**
- For tasks requiring high-quality, deterministic outputs
- When you need multiple diverse but high-probability sequences
- In scenarios where finding the most likely sequence is important
- For applications where you can afford the computational cost 