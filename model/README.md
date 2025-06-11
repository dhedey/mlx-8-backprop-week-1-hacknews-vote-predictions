# PostgreSQL DataLoader for HackerNews Data

This directory contains PyTorch DataLoader implementations that connect to a PostgreSQL database to load HackerNews data for machine learning training.

## Files

- `data_loader.py` - Original PostgreSQL DataLoader implementation (loads all data into memory)
- `streaming_data_loader.py` - **NEW**: Streaming PostgreSQL DataLoader (loads data on-demand)
- `train_with_db.py` - Example training script using the streaming database DataLoader
- `config.py` - Configuration management for database and model parameters

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

### Streaming DataLoader Usage (Recommended)

```python
from streaming_data_loader import HackerNewsStreamingDataLoader, create_connection_params

# Create connection parameters
connection_params = create_connection_params()  # Uses environment variables

# Create streaming data loader
data_loader = HackerNewsStreamingDataLoader(
    connection_params=connection_params,
    table_name='hacker_news.items_by_month',
    columns=['id', 'title', 'score'],
    filter_condition=None,  # Uses default HackerNews filter
    train_split=0.8,        # 80% training (ID % 10 < 8)
    batch_size=32,
    db_batch_size=1000      # Fetch 1000 rows per DB query
)

# Get train and test loaders - starts immediately, no waiting!
train_loader = data_loader.get_train_loader()
test_loader = data_loader.get_test_loader()
```

### Train/Test Split Method

The streaming loader uses **ID modulo 10** for deterministic splits:
- **Training**: `ID % 10 < 8` (80% of data)  
- **Testing**: `ID % 10 >= 8` (20% of data)

This ensures:
- ✅ Consistent splits across runs
- ✅ No random state dependency  
- ✅ Chronological ordering within each split

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

### Streaming PostgreSQL DataLoader (Recommended)

**NEW**: `streaming_data_loader.py` provides memory-efficient data loading:

- **On-demand loading**: Fetches data in batches as needed (no 4-hour wait!)
- **Deterministic train/test split**: Uses `ID % 10` for consistent splits
- **Time-ordered data**: Orders by `time` column for chronological training
- **Memory efficient**: Only loads what's needed for current batch
- **Progress tracking**: Shows loading progress with database batch fetching

### Original PostgreSQL DataLoader

`data_loader.py` (legacy approach):
- Loads entire dataset into memory
- Uses random train/test split
- Better for smaller datasets
- May take hours for large datasets

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
