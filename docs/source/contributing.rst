Contributing
============

We welcome contributions to the LLM Samplers project! Here's how you can help:

Development Setup
---------------

1. Fork the repository on GitHub.
2. Clone your fork locally:

   .. code-block:: bash

       git clone https://github.com/your-username/llm-samplers.git
       cd llm-samplers

3. Create a virtual environment and install development dependencies:

   .. code-block:: bash

       python -m venv .venv
       source .venv/bin/activate  # On Unix/macOS
       # or
       .venv\Scripts\activate  # On Windows
       
       pip install -e ".[dev]"

4. Set up pre-commit hooks:

   .. code-block:: bash

       pre-commit install

Code Style
---------

This project uses Ruff for linting and formatting. Before submitting a pull request:

.. code-block:: bash

    # Check for issues
    ruff check .
    
    # Format your code
    ruff format .

All code should follow the project's style guidelines, which are enforced by Ruff.

Writing Tests
-----------

We use pytest for testing. Please include tests for any new features or bug fixes:

1. Add test files in the ``tests/`` directory
2. Run the tests:

   .. code-block:: bash

       python -m pytest tests/
       
       # For more verbose output
       python -m pytest tests/ -v

Pull Request Process
------------------

1. Update the documentation if necessary.
2. Make sure all tests pass.
3. Update the README.md if needed.
4. Submit a pull request with a clear description of the changes.

Creating a New Sampler
--------------------

If you're implementing a new sampling technique:

1. Create a new file in ``src/llm_samplers/`` named after your sampler (e.g., ``my_sampler.py``)
2. Extend the base ``Sampler`` class from ``base.py``
3. Implement the required methods:
   - ``__init__``: Initialize your sampler with appropriate parameters
   - ``adjust_logits``: The main method that modifies the logits
   - Any additional helper methods you need
4. Add your sampler to ``__init__.py``
5. Create tests in the ``tests/`` directory
6. Update documentation to include your new sampler

Documentation
------------

To build the documentation locally:

.. code-block:: bash

    cd docs
    pip install -r requirements.txt
    sphinx-build -b html source _build/html

Then open ``_build/html/index.html`` in your browser to view the documentation. 