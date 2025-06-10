# Set-up

Install the `uv` package manager.

To avoid an issue when running Jupyter notebook with scipy, you may need to run:

```bash
uv pip install pip setuptools
```

Then install dependencies with:

```bash
uv sync --all-packages --dev
```

