Installation
============

From PyPI
--------

The recommended way to install LLM Samplers is via pip:

.. code-block:: bash

    pip install llm-samplers

From Source
----------

To install LLM Samplers from source:

1. Clone the repository:

.. code-block:: bash

    git clone https://github.com/iantimmis/llm-samplers.git
    cd llm-samplers

2. Create and activate a virtual environment (recommended):

.. code-block:: bash

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

3. Install the package in development mode:

.. code-block:: bash

    # Using pip
    pip install -e ".[dev]"  # Includes development dependencies
    
    # Using uv (recommended)
    uv pip install -e .  # uv installs dev dependencies by default

Requirements
-----------

LLM Samplers requires Python 3.9 or later and has the following dependencies:

* torch >= 2.0.0
* transformers >= 4.30.0
* numpy >= 1.24.0 