# Resumable Batch Preprocessor for HackerNews Data

This script streams data from a PostgreSQL source database, applies preprocessing transformations, and writes the results to a target database using efficient cursor-based pagination.

## 🚀 Key Features

- **Cursor-based pagination**: Uses `(time, id)` cursors for efficient streaming (no slow OFFSET queries)
- **Resumable processing**: Automatically resumes from where it left off if interrupted
- **Auto database creation**: Creates target database and schemas if they don't exist
- **Progress tracking**: Real-time progress with persistent state in database
- **Configurable batch processing**: Process data in customizable batch sizes
- **Data validation and cleaning**: Removes invalid records, normalizes text
- **Text preprocessing**: Tokenization, feature extraction, keyword detection
- **Feature engineering**: Time-based features, score transformations
- **Error handling**: Comprehensive logging and error recovery
- **Background processing**: Can be stopped with Ctrl+C and resumed later

## Installation

Install dependencies:
```bash
# Add to pyproject.toml dependencies
uv add pandas numpy sqlalchemy psycopg2-binary tqdm python-dotenv
```

## Configuration

Set up environment variables for database connections:

### Single Database (Source and Target same)
```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=your_database
export POSTGRES_USER=your_user
export POSTGRES_PASSWORD=your_password
```

### Separate Source and Target Databases
```bash
# Source database
export SOURCE_POSTGRES_HOST=source_host
export SOURCE_POSTGRES_PORT=5432
export SOURCE_POSTGRES_DB=source_db
export SOURCE_POSTGRES_USER=source_user
export SOURCE_POSTGRES_PASSWORD=source_password

# Target database
export TARGET_POSTGRES_HOST=target_host
export TARGET_POSTGRES_PORT=5432
export TARGET_POSTGRES_DB=target_db
export TARGET_POSTGRES_USER=target_user
export TARGET_POSTGRES_PASSWORD=target_password
```

## Quick Start

### 1. Basic Usage (Auto-Resume Enabled)
```bash
# Process all HackerNews stories with automatic resuming
python batch_preprocessor.py

# Test with limited data
python batch_preprocessor.py --limit 1000

# Use custom batch size
python batch_preprocessor.py --batch-size 2000
```

### 2. Resume from Interruption
If processing is interrupted (Ctrl+C or error), simply run the same command again:
```bash
# This will automatically resume from where it left off
python batch_preprocessor.py
```

### 3. Manual Resume from Specific Position
```bash
python batch_preprocessor.py \
    --resume-time "2023-01-15 10:30:00" \
    --resume-id 12345 \
    --no-auto-resume
```

### 4. Check Processing Status
```bash
# Show current progress and history
python example_usage.py --show-progress

# Test database connection
python example_usage.py --test-connection
```

## Source Query

The script processes data matching this filter:
```sql
SELECT
    id, title, score, time, type, url, dead
FROM hacker_news.items
WHERE
    type = 'story'
    AND title IS NOT NULL
    AND url IS NOT NULL
    AND score IS NOT NULL AND score >= 1
    AND (dead IS NULL OR dead = false)
ORDER BY time, id
```

## Preprocessing Features

### Data Validation
- Removes records with invalid scores (< 1)
- Filters out null/empty titles
- Normalizes whitespace in text

### Text Features
- `title_normalized`: Lowercase normalized title
- `title_length`: Character count
- `title_word_count`: Word count
- `has_question`: Contains question mark
- `has_exclamation`: Contains exclamation point
- `has_numbers`: Contains numeric characters
- Tech keyword flags (AI, ML, Python, JavaScript, etc.)

### Time Features
- `hour_of_day`: Hour (0-23)
- `day_of_week`: Day of week (0=Monday)
- `day_of_month`: Day of month
- `month`: Month
- `year`: Year
- `is_weekend`: Weekend indicator
- `is_business_hours`: Business hours indicator

### Score Features
- `log_score`: Natural logarithm of score
- `score_category`: Categorical score ranges (low, medium, high, very_high, viral)
- `score_zscore`: Z-score normalized within batch

## Target Database Structure

The script automatically creates:

### Main Table: `processed.hackernews_items`
Contains all preprocessed data with original columns plus:
- All original columns (`id`, `title`, `score`, `time`, etc.)
- `processed_at`: Timestamp of processing
- Text features: `title_normalized`, `title_length`, `title_word_count`, etc.
- Time features: `hour_of_day`, `day_of_week`, `is_weekend`, etc.
- Score features: `log_score`, `score_category`, `score_zscore`

### Progress Table: `processed.hackernews_items_progress`
Tracks processing state for resumability:
- `source_table`, `target_table`: Table identifiers
- `last_processed_time`, `last_processed_id`: Cursor position
- `total_processed`: Count of processed records
- `status`: 'running', 'completed', or 'failed'
- `started_at`, `updated_at`: Timestamps

## Resumable Processing

### How It Works
1. **Progress Tracking**: Each batch updates the progress table with cursor position
2. **Automatic Resume**: On restart, checks progress table for last cursor position
3. **Interrupt Safety**: Ctrl+C saves current position before stopping
4. **Error Recovery**: Failed runs can be resumed from last successful batch

### Resume Behavior
- **Fresh Start**: No progress record → starts from beginning
- **Auto Resume**: Progress record exists → continues from last cursor
- **Manual Override**: `--resume-time` and `--resume-id` override auto-resume
- **Disable Resume**: `--no-auto-resume` always starts fresh

## Performance

- **Cursor-based pagination**: Eliminates slow OFFSET queries
- **Batch processing**: Configurable batch sizes (default: 1000 records)
- **Memory efficient**: Processes data in streaming fashion
- **Resumable**: Can restart from any cursor position

## Logging

Logs are written to both console and `preprocessing.log` file:
- Progress tracking with record counts
- Cursor positions for resumability
- Error details and stack traces
- Performance metrics

## Example Output

```
2025-06-11 10:15:30 - Starting batch preprocessing pipeline
2025-06-11 10:15:30 - Source: source_db.hacker_news.items
2025-06-11 10:15:30 - Target: target_db.processed.hackernews_items
2025-06-11 10:15:31 - Total records to process: 1,234,567
2025-06-11 10:15:32 - Created target table: processed.hackernews_items
Processing batches: 45%|████▌     | 567890/1234567 [02:15<02:43, 4093.21it/s]
2025-06-11 10:17:45 - Batch 568: Processed 1000 rows (Total: 567,890) - Cursor: (2023-03-15 14:22:33, 45678)
```

## Error Recovery

If processing fails, note the last cursor position from logs and resume:
```bash
python batch_preprocessor.py \
    --resume-time "2023-03-15 14:22:33" \
    --resume-id 45678
```
