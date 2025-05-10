# LLM Samplers Documentation

This directory contains the documentation for the LLM Samplers project, built with Sphinx and hosted on ReadTheDocs.

## Building the Documentation Locally

### Install Requirements

```bash
# From the docs directory
pip install -r requirements.txt
```

### Build HTML Documentation

```bash
# From the docs directory
sphinx-build -b html source _build/html
```

Or if you prefer using the Makefile:

```bash
# From the docs directory
make html
```

The built documentation will be in the `_build/html` directory, and you can open `_build/html/index.html` in your browser.

## Hosting on ReadTheDocs

The documentation is set up to be hosted on [ReadTheDocs](https://readthedocs.org). To set up hosting:

1. Go to https://readthedocs.org and create an account (if you don't have one already)
2. Import your GitHub repository
3. The configuration is already set up in the `.readthedocs.yaml` file at the root of the repository

ReadTheDocs will automatically build the documentation whenever you push to the repository.

## Documentation Structure

- `source/`: Contains the source files for the documentation
  - `conf.py`: Configuration file for Sphinx
  - `index.rst`: Main index file
  - Other RST files for different sections
- `requirements.txt`: Dependencies needed to build the docs
- `Makefile`: Helps with building the documentation
- `_static/`: Directory for static files (CSS, images, etc.)
- `_build/`: Build output directory (created when building the docs)

## Adding New Content

To add new content to the documentation:

1. Create a new `.rst` file in the `source/` directory
2. Add it to the table of contents in `index.rst` or another relevant file
3. Build the documentation to see your changes
