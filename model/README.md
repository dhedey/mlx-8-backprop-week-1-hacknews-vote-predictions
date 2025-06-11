# PostgreSQL DataLoader for HackerNews Data

This directory contains PyTorch DataLoader implementations that connect to a PostgreSQL database to load HackerNews data for machine learning training.

## Files

- `data_loader.py` - Main PostgreSQL DataLoader implementation
- `train_with_db.py` - Example training script using the database DataLoader
- `config.py` - Configuration management for database and model parameters
- `model.py` - Original model architecture (MNIST-based)

## Requirements

Make sure you have the required dependencies installed:

```bash
uv sync  # This will install dependencies from pyproject.toml
```

Or manually install:
```bash
pip install psycopg2-binary pandas scikit-learn torch
```

## Database Setup

### Environment Variables

Set up your database connection using environment variables:

```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=your_database_name
export POSTGRES_USER=your_username
export POSTGRES_PASSWORD=your_password
```

### Table Schema

The DataLoader expects a table named `items_by_month` with at least the following columns:
- `id` - Unique identifier
- `title` - Text title of the item
- `dead` - Boolean flag (nullable)

Example table creation:
```sql
CREATE TABLE items_by_month (
    id SERIAL PRIMARY KEY,
    title TEXT,
    dead BOOLEAN,
    score INTEGER,
    time TIMESTAMP
    -- other columns as needed
);
```

## Usage

### Basic Usage

```python
from data_loader import HackerNewsDataLoader, create_connection_params

# Create connection parameters
connection_params = create_connection_params(
    host='localhost',
    port=5432,
    database='your_db',
    user='your_user',
    password='your_password'
)

# Create data loader
data_loader = HackerNewsDataLoader(
    connection_params=connection_params,
    table_name='items_by_month',
    columns=['id', 'title'],
    filter_condition="(dead IS NULL OR dead = false)",
    train_split=0.8,
    batch_size=32
)

# Get train and test loaders
train_loader = data_loader.get_train_loader()
test_loader = data_loader.get_test_loader()
```

### Training Example

```python
# Run the complete training example
python train_with_db.py
```

### Custom Configuration

```python
from config import Config

# Use configuration class
db_params = Config.get_db_params()
model_params = Config.get_model_params()
training_params = Config.get_training_params()
```

## Features

### PostgreSQLDataset Class

- Connects to PostgreSQL database using psycopg2
- Loads data with customizable SQL queries
- Supports filtering with WHERE clauses
- Returns data as PyTorch-compatible dictionaries

### HackerNewsDataLoader Class

- Automatic train/test split (default 80/20)
- Configurable batch size and shuffling
- Reproducible random splits
- Memory-efficient data loading

### Key Parameters

- `train_split`: Fraction of data for training (default: 0.8)
- `batch_size`: Number of samples per batch (default: 32)
- `shuffle`: Whether to shuffle training data (default: True)
- `random_state`: Seed for reproducibility (default: 42)

## SQL Query Structure

The DataLoader builds queries in this format:

```sql
SELECT id, title
FROM items_by_month
WHERE (dead IS NULL OR dead = false)
ORDER BY id
```

You can customize:
- Column selection with `columns` parameter
- Table name with `table_name` parameter
- Filter conditions with `filter_condition` parameter

## Error Handling

The DataLoader includes error handling for:
- Database connection failures
- Invalid SQL queries
- Missing tables or columns
- Network connectivity issues

## Performance Considerations

- Uses `num_workers=0` for database connections to avoid connection pool issues
- Loads entire dataset into memory (consider pagination for very large datasets)
- Connection is closed after data loading to prevent connection leaks

## Extending the DataLoader

You can extend the DataLoader for specific use cases:

```python
class CustomHackerNewsDataLoader(HackerNewsDataLoader):
    def __init__(self, *args, **kwargs):
        # Add custom columns or filters
        kwargs['columns'] = ['id', 'title', 'score', 'url']
        kwargs['filter_condition'] = "score > 10 AND (dead IS NULL OR dead = false)"
        super().__init__(*args, **kwargs)
```

## Troubleshooting

### Common Issues

1. **Connection Error**: Check database credentials and network connectivity
2. **Table Not Found**: Verify table name and schema
3. **Column Not Found**: Check column names match database schema
4. **Memory Issues**: Consider reducing batch size for large datasets

### Debug Mode

Enable verbose output by checking the console output when creating the DataLoader. It will print:
- The SQL query being executed
- Number of rows loaded
- Dataset split information

## Example Output

```
Executing query: SELECT id, title FROM items_by_month WHERE (dead IS NULL OR dead = false) ORDER BY id
Loaded 10000 rows from items_by_month
Dataset split - Train: 8000, Test: 2000
Dataset Info:
  total_samples: 10000
  train_samples: 8000
  test_samples: 2000
  columns: ['id', 'title']
  table_name: items_by_month
  filter_condition: (dead IS NULL OR dead = false)
```
