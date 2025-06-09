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

### urls.csv (Missing - Git LFS Setup)

The large `urls.csv` file (~375MB, 4.6M URLs) was removed during Git LFS setup to avoid GitHub's file size limits.

**To restore the file:**

1. **If you have the original data source**, place the CSV file here with columns:
   ```csv
   id,url
   18385172,https://example.com/article
   ```

2. **Add it through Git LFS:**
   ```bash
   # File should be automatically tracked by LFS (*.csv pattern is configured)
   git add analysis/urls.csv
   git commit -m "Add large URLs dataset via Git LFS"
   git push
   ```

3. **Run domain analysis:**
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

The domain analysis script has already been run and produced:
- **412,898 unique domains** from the dataset
- **Top domains**: github.com (3.67%), medium.com (2.84%), youtube.com (2.62%)
- Results saved in `main_domain_analysis.txt` 