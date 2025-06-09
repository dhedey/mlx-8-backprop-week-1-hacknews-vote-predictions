# Analysis Directory

This directory contains analysis results and SQL queries for the Hacker News upvote prediction project.

## Files

### SQL Queries
- `analyse-data-quality.sql` - Data quality analysis
- `select-items.sql` - Basic item selection
- `summary-by-author.sql` - Author-based summaries
- `summary-by-day-hour.sql` - Time-based analysis
- `summary-counts-per-month.sql` - Monthly count statistics
- `summary-scores-for-single-author.sql` - Individual author analysis
- `summary-scores.sql` - Score summaries
- `valid-sample-data.sql` - Sample data validation

### Analysis Results
- `main_domain_analysis.txt` - Domain analysis results (24MB)

## Large Dataset Files

### urls.csv ✅ Available (Git LFS)

The large `urls.csv` file (358MB, 4.6M URLs) is now available and tracked via Git Large File Storage.

**File Details:**
- **Size**: 358MB 
- **Records**: ~4.6 million URLs
- **Format**: CSV with columns `id,url`
- **Storage**: Git LFS (Large File Storage)

**Sample data:**
```csv
id,url
18385172,https://www.scientificamerican.com/article/...
18385182,https://usehooks.com/
21978979,https://www.nytimes.com/2020/01/07/nyregion/...
```

**To run domain analysis:**
```bash
cd scripts
python3 domain_analysis.py ../analysis/urls.csv
```

## Git LFS Configuration

This repository is configured to track `*.csv` files through Git Large File Storage (LFS):
- `.gitattributes` contains: `*.csv filter=lfs diff=lfs merge=lfs -text`
- Large CSV files are automatically handled by LFS
- GitHub's 100MB file limit is bypassed for LFS-tracked files

## Domain Analysis Results

The domain analysis script has been run and produced:
- **412,898 unique domains** from the dataset
- **Top domains**: github.com (3.67%), medium.com (2.84%), youtube.com (2.62%)
- Results saved in `main_domain_analysis.txt`

## Usage Examples

**Run domain analysis:**
```bash
# From project root
python3 scripts/domain_analysis.py analysis/urls.csv

# With custom output directory
python3 scripts/domain_analysis.py analysis/urls.csv --output results/

# Show examples only
python3 scripts/domain_analysis.py --examples
``` 