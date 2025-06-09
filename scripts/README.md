# Domain Analysis Script

Python script to analyze URLs from training datasets and extract main domain statistics with enhanced command-line interface and flexible file handling.

## Features

- ✅ Extracts main domains from URLs (ignores subdomains)
- ✅ Handles complex TLDs correctly (e.g., `.co.uk`, `.com.au`)
- ✅ Processes large CSV files efficiently
- ✅ Exports results to multiple formats (TXT, CSV)
- ✅ Progress tracking for large datasets
- ✅ Smart path handling (relative/absolute paths)
- ✅ Professional command-line interface with argument parsing
- ✅ File validation and error handling
- ✅ Flexible output directory options

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Alternative installation (conda):**
   ```bash
   conda install -c conda-forge tldextract
   ```

## Usage

### Command-line Syntax
```bash
python3 domain_analysis.py [OPTIONS] CSV_FILE
```

### Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `CSV_FILE` | Yes* | Path to CSV file containing URLs (must have "url" column) |
| `-o, --output DIR` | No | Output directory for results (default: same as input file) |
| `--examples` | No | Show domain extraction examples and exit |
| `-h, --help` | No | Show help message and exit |

*Required unless using `--examples`

### Basic Usage Examples

**1. Process CSV file (saves results next to input file):**
```bash
python3 domain_analysis.py data/urls.csv
```

**2. Process with custom output directory:**
```bash
python3 domain_analysis.py data/urls.csv --output results/
```

**3. Process from different directory:**
```bash
python3 domain_analysis.py ../data/urls.csv
```

**4. Use absolute paths:**
```bash
python3 domain_analysis.py /full/path/to/urls.csv --output /full/path/to/results/
```

**5. Show examples only:**
```bash
python3 domain_analysis.py --examples
```

**6. Get help:**
```bash
python3 domain_analysis.py --help
```

### Expected CSV Format

Your CSV file should have a header row with a `url` column:
```csv
id,url
1,https://github.com/user/repo
2,https://medium.com/article
3,https://subdomain.example.com/page
```

## File Organization

### Input/Output Behavior

- **Default behavior**: Results are saved in the same directory as the input CSV file
- **Custom output**: Use `--output` to specify a different directory
- **Auto-creation**: Output directories are created automatically if they don't exist

### Example File Organization
```
project/
├── data/
│   ├── urls.csv                     # Input file
│   ├── domain_frequencies.csv       # Generated output
│   ├── main_domain_analysis.txt     # Generated output
│   └── domain_analysis_summary.csv  # Generated output
└── scripts/
    └── domain_analysis.py           # Script location
```

## Output Files

The script generates three output files in the specified output directory:

1. **`main_domain_analysis.txt`** - Human-readable report with statistics
2. **`domain_frequencies.csv`** - All domains with frequencies and percentages  
3. **`domain_analysis_summary.csv`** - Key statistics summary

## Example Output

### Console Output
```
📁 Input file: /full/path/to/data/urls.csv
📁 Output directory: /full/path/to/data
Processing urls.csv...
Processed 50,000 URLs...
...
Top 20 most frequent main domains:
--------------------------------------------------
github.com                      169,484 ( 3.67%)
medium.com                      130,879 ( 2.84%)
youtube.com                     121,080 ( 2.62%)
nytimes.com                      80,460 ( 1.74%)
blogspot.com                     61,027 ( 1.32%)

📄 Complete results saved to: /path/to/main_domain_analysis.txt
📊 CSV export saved to: /path/to/domain_frequencies.csv
📋 Summary CSV saved to: /path/to/domain_analysis_summary.csv
```

### CSV Format (`domain_frequencies.csv`)
```csv
domain,frequency,percentage
github.com,169484,3.6725
medium.com,130879,2.8359
youtube.com,121080,2.6236
```

### Summary CSV (`domain_analysis_summary.csv`)
```csv
metric,value
input_file,/full/path/to/data/urls.csv
total_urls_processed,4616701
valid_domains_extracted,4615003
unique_domains_found,412898
top_domain,github.com
top_domain_count,169484
```

## Domain Consolidation Examples

The script consolidates subdomains to main domains:

- `googleblog.blogspot.com` → `blogspot.com`
- `en.wikipedia.org` → `wikipedia.org`
- `mail.google.com` → `google.com`
- `docs.google.com` → `google.com`
- `user.github.io` → `github.io`

## Error Handling

The script includes robust error handling:

- ✅ **File validation**: Checks if CSV file exists and is readable
- ✅ **Format validation**: Verifies CSV has required 'url' column
- ✅ **Path resolution**: Handles relative and absolute paths correctly
- ✅ **Clear error messages**: Shows current directory and helpful suggestions

### Error Examples
```bash
❌ Error: File 'missing.csv' does not exist.
Current working directory: /Users/example/project
Please check the file path and try again.

❌ Error: CSV file must have a 'url' column.
Found columns: ['id', 'link', 'title']
```

## Advanced Usage

### Integration Examples

**From project root:**
```bash
python3 1-hackernews-upvote-prediction/scripts/domain_analysis.py data/urls.csv
```

**With custom output for organization:**
```bash
python3 scripts/domain_analysis.py data/training.csv --output analysis/domains/
```

**Batch processing:**
```bash
for file in data/*.csv; do
    python3 scripts/domain_analysis.py "$file" --output "results/$(basename "$file" .csv)/"
done
```

## Dependencies

- **Required:** `tldextract>=5.0.0` - For accurate domain extraction
- **Built-in:** `csv`, `urllib.parse`, `collections`, `sys`, `os`, `argparse`

## Performance

- Processes ~4.6M URLs in approximately 2-3 minutes
- Memory efficient with chunked processing
- Progress indicators for large files (every 50K URLs)
- Automatic directory creation for outputs

## Troubleshooting

**File not found?**
- Check your current directory with `pwd`
- Use absolute paths if unsure
- Verify the CSV file exists with `ls -la`

**Permission errors?**
- Ensure write permissions to output directory
- Use `--output` to specify a writable location

**Missing 'url' column?**
- Check CSV headers match expected format
- Ensure first row contains column names

## License

This script is provided as-is for data analysis purposes. 