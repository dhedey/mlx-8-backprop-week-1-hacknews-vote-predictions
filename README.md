# Set-up

Install the `uv` package manager.

To avoid an issue when running Jupyter notebook with scipy, you may need to run:

Then add your local configuration to a `.env` file:
```bash
POSTGRES_HOST=??
POSTGRES_PORT=5432
POSTGRES_DB=??
POSTGRES_USER=??
POSTGRES_PASSWORD=??
```

Then install dependencies with:

```bash
uv sync --all-packages --dev
```

Run model training with:
```bash
uv run ./model/train.py
```
