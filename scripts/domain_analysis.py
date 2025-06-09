#!/usr/bin/env python3
"""
Script to analyze main domains from training dataset (ignoring subdomains):
1. Extract all URLs from the CSV file
2. Parse each URL to get the main domain only (e.g., googleblog.blogspot.com -> blogspot.com)
3. Count occurrences of each main domain
4. Display results sorted by frequency
5. Export results to both text report and CSV file
"""

import csv
from urllib.parse import urlparse
from collections import Counter
import sys
import os
import argparse

# Try to import tldextract for proper domain extraction
try:
    import tldextract
    HAS_TLDEXTRACT = True
    print("Using tldextract library for accurate domain extraction")
except ImportError:
    HAS_TLDEXTRACT = False
    print("tldextract not available, using heuristic approach")
    print("For better accuracy, install with: pip install tldextract")

def extract_main_domain_tldextract(url):
    """Extract main domain using tldextract library (most accurate)"""
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        extracted = tldextract.extract(url)
        
        # Return domain.suffix (e.g., 'google.com', 'blogspot.com')
        if extracted.domain and extracted.suffix:
            return f"{extracted.domain}.{extracted.suffix}".lower()
        return None
    except Exception as e:
        print(f"Error parsing URL '{url}': {e}")
        return None

def extract_main_domain_heuristic(url):
    """Extract main domain using simple heuristic (fallback method)"""
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Remove www. prefix
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Split by dots and take last 2 parts for most cases
        parts = domain.split('.')
        if len(parts) >= 2:
            # Handle special cases like .co.uk, .com.au, etc.
            if len(parts) >= 3 and parts[-2] in ['co', 'com', 'net', 'org', 'gov', 'edu', 'ac']:
                return '.'.join(parts[-3:])
            else:
                return '.'.join(parts[-2:])
        
        return domain
    except Exception as e:
        print(f"Error parsing URL '{url}': {e}")
        return None

def extract_main_domain(url):
    """Extract main domain using the best available method"""
    if HAS_TLDEXTRACT:
        return extract_main_domain_tldextract(url)
    else:
        return extract_main_domain_heuristic(url)

def validate_csv_file(csv_file_path):
    """Validate that the CSV file exists and is readable"""
    if not os.path.exists(csv_file_path):
        print(f"❌ Error: File '{csv_file_path}' does not exist.")
        print(f"Current working directory: {os.getcwd()}")
        print("Please check the file path and try again.")
        return False
    
    if not os.path.isfile(csv_file_path):
        print(f"❌ Error: '{csv_file_path}' is not a file.")
        return False
    
    try:
        # Test if file can be opened and has valid CSV structure
        with open(csv_file_path, 'r', encoding='utf-8', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            headers = reader.fieldnames
            if not headers or 'url' not in headers:
                print(f"❌ Error: CSV file must have a 'url' column.")
                print(f"Found columns: {headers}")
                return False
        return True
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        return False

def resolve_output_path(csv_file_path, output_filename):
    """Generate output file path in the same directory as input CSV"""
    csv_dir = os.path.dirname(os.path.abspath(csv_file_path))
    return os.path.join(csv_dir, output_filename)

def analyze_main_domains(csv_file_path, output_dir=None):
    """Analyze main domains from CSV file containing URLs"""
    
    # Validate input file
    if not validate_csv_file(csv_file_path):
        return None
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(csv_file_path))
    elif not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    print(f"📁 Input file: {os.path.abspath(csv_file_path)}")
    print(f"📁 Output directory: {os.path.abspath(output_dir)}")
    print(f"Processing {os.path.basename(csv_file_path)}...")
    
    domains = []
    total_urls = 0
    invalid_urls = 0
    
    try:
        with open(csv_file_path, 'r', encoding='utf-8', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                total_urls += 1
                url = row.get('url', '').strip()
                
                if not url:
                    invalid_urls += 1
                    continue
                    
                main_domain = extract_main_domain(url)
                if main_domain:
                    domains.append(main_domain)
                else:
                    invalid_urls += 1
                    
                # Progress indicator for large files
                if total_urls % 50000 == 0:
                    print(f"Processed {total_urls:,} URLs...")
    
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
    print(f"\nProcessing complete!")
    print(f"Total URLs processed: {total_urls:,}")
    print(f"Valid main domains extracted: {len(domains):,}")
    print(f"Invalid/unparseable URLs: {invalid_urls:,}")
    
    # Count domain occurrences
    domain_counts = Counter(domains)
    unique_domains = len(domain_counts)
    
    print(f"Unique main domains found: {unique_domains:,}")
    print(f"\nTop 20 most frequent main domains:")
    print("-" * 50)
    
    for domain, count in domain_counts.most_common(20):
        percentage = (count / len(domains)) * 100
        print(f"{domain:<30} {count:>8,} ({percentage:>5.2f}%)")
    
    # Save results to text file
    output_file = os.path.join(output_dir, "main_domain_analysis.txt")
    with open(output_file, 'w') as f:
        f.write(f"Main Domain Analysis Results\n")
        f.write(f"===============================\n\n")
        f.write(f"Input file: {os.path.abspath(csv_file_path)}\n")
        f.write(f"Method used: {'tldextract library' if HAS_TLDEXTRACT else 'heuristic approach'}\n")
        f.write(f"Total URLs processed: {total_urls:,}\n")
        f.write(f"Valid main domains extracted: {len(domains):,}\n")
        f.write(f"Invalid/unparseable URLs: {invalid_urls:,}\n")
        f.write(f"Unique main domains found: {unique_domains:,}\n\n")
        f.write(f"All main domains sorted by frequency:\n")
        f.write("-" * 50 + "\n")
        
        for domain, count in domain_counts.most_common():
            percentage = (count / len(domains)) * 100
            f.write(f"{domain:<40} {count:>8,} ({percentage:>6.2f}%)\n")
    
    print(f"\n📄 Complete results saved to: {output_file}")
    
    # Save results to CSV file
    csv_output_file = os.path.join(output_dir, "domain_frequencies.csv")
    with open(csv_output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['domain', 'frequency', 'percentage'])
        
        # Write data rows
        for domain, count in domain_counts.most_common():
            percentage = (count / len(domains)) * 100
            writer.writerow([domain, count, f"{percentage:.4f}"])
    
    print(f"📊 CSV export saved to: {csv_output_file}")
    
    # Also create a summary CSV with key statistics
    summary_csv_file = os.path.join(output_dir, "domain_analysis_summary.csv")
    with open(summary_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write summary statistics
        writer.writerow(['metric', 'value'])
        writer.writerow(['input_file', os.path.abspath(csv_file_path)])
        writer.writerow(['total_urls_processed', total_urls])
        writer.writerow(['valid_domains_extracted', len(domains)])
        writer.writerow(['invalid_urls', invalid_urls])
        writer.writerow(['unique_domains_found', unique_domains])
        writer.writerow(['method_used', 'tldextract library' if HAS_TLDEXTRACT else 'heuristic approach'])
        writer.writerow(['top_domain', domain_counts.most_common(1)[0][0] if domain_counts else 'N/A'])
        writer.writerow(['top_domain_count', domain_counts.most_common(1)[0][1] if domain_counts else 0])
    
    print(f"📋 Summary CSV saved to: {summary_csv_file}")
    
    return domain_counts

def show_examples():
    """Show some examples of domain extraction"""
    test_urls = [
        "https://googleblog.blogspot.com/2021/01/example.html",
        "https://subdomain.github.com/user/repo",
        "https://www.bbc.co.uk/news/technology",
        "https://mail.google.com/inbox",
        "https://docs.google.com/document",
        "https://en.wikipedia.org/wiki/Python",
        "https://stackoverflow.com/questions/123456"
    ]
    
    print("\nExample domain extractions:")
    print("-" * 60)
    for url in test_urls:
        main_domain = extract_main_domain(url)
        print(f"{url:<45} -> {main_domain}")
    print()

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Analyze main domains from URL dataset',
        epilog="""
Examples:
  %(prog)s data/urls.csv                    # Process CSV file
  %(prog)s ../data/urls.csv --output results/  # Custom output directory
  %(prog)s /full/path/to/urls.csv          # Absolute path
  %(prog)s --examples                      # Show examples only
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('csv_file', nargs='?',
                       help='Path to CSV file containing URLs (must have "url" column)')
    parser.add_argument('-o', '--output', 
                       help='Output directory for results (default: same as input file)',
                       metavar='DIR')
    parser.add_argument('--examples', 
                       action='store_true',
                       help='Show domain extraction examples and exit')
    
    args = parser.parse_args()
    
    if args.examples:
        show_examples()
        return
    
    if not args.csv_file:
        parser.error("csv_file is required unless using --examples")
    
    # Show examples first
    show_examples()
    
    try:
        domain_counts = analyze_main_domains(args.csv_file, args.output)
        if domain_counts is None:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 