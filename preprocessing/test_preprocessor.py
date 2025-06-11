#!/usr/bin/env python3
"""
Quick test of the batch preprocessor with source database only.
This demonstrates the cursor-based pagination and preprocessing features.
"""

import os
import sys
from batch_preprocessor import BatchPreprocessor, DatabaseConfig


def test_source_connection():
    """Test connection to source database only."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    # Test source database
    source_config = DatabaseConfig.from_env("SOURCE")
    
    print("Testing SOURCE database connection...")
    print(f"Host: {source_config.host}")
    print(f"Port: {source_config.port}")
    print(f"Database: {source_config.database}")
    print(f"User: {source_config.user}")
    
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(source_config.get_connection_url())
        
        with engine.connect() as conn:
            # Test basic connection
            result = conn.execute(text("SELECT version()"))
            version = result.scalar()
            print(f"✅ SOURCE connection successful!")
            print(f"PostgreSQL version: {version}")
            
            # Test source table and count matching records
            result = conn.execute(text("""
                SELECT COUNT(*) 
                FROM hacker_news.items_by_month 
                WHERE type = 'story'
                  AND title IS NOT NULL
                  AND url IS NOT NULL
                  AND score IS NOT NULL AND score >= 1
                  AND (dead IS NULL OR dead = false)
            """))
            
            count = result.scalar()
            print(f"✅ Source table accessible: {count:,} matching records found")
            
            # Test cursor bounds
            result = conn.execute(text("""
                SELECT 
                    MIN(time) as min_time,
                    MAX(time) as max_time,
                    MIN(id) as min_id,
                    MAX(id) as max_id
                FROM hacker_news.items_by_month 
                WHERE type = 'story'
                  AND title IS NOT NULL
                  AND url IS NOT NULL
                  AND score IS NOT NULL AND score >= 1
                  AND (dead IS NULL OR dead = false)
            """))
            
            row = result.fetchone()
            print(f"📅 Time range: {row[0]} to {row[1]}")
            print(f"🔢 ID range: {row[2]:,} to {row[3]:,}")
            
            # Test a small sample
            result = conn.execute(text("""
                SELECT id, title, score, time
                FROM hacker_news.items_by_month 
                WHERE type = 'story'
                  AND title IS NOT NULL
                  AND url IS NOT NULL
                  AND score IS NOT NULL AND score >= 1
                  AND (dead IS NULL OR dead = false)
                ORDER BY time, id
                LIMIT 3
            """))
            
            print("\n📋 Sample records:")
            for row in result.fetchall():
                print(f"  ID: {row[0]}, Score: {row[2]}, Time: {row[3]}")
                print(f"  Title: {row[1][:60]}...")
                print()
            
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"❌ SOURCE connection failed: {e}")
        return False


def test_preprocessing_dry_run():
    """Test preprocessing on a small sample without writing to target."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    # Only test source connection
    source_config = DatabaseConfig.from_env("SOURCE")
    
    print("\n🧪 Testing preprocessing logic (dry run)...")
    
    try:
        from sqlalchemy import create_engine, text
        import pandas as pd
        from batch_preprocessor import DataPreprocessor
        
        engine = create_engine(source_config.get_connection_url())
        preprocessor = DataPreprocessor()
        
        # Get a small sample
        with engine.connect() as conn:
            query = """
                SELECT id, title, score, time, type, url, dead, by
                FROM hacker_news.items_by_month 
                WHERE type = 'story'
                  AND title IS NOT NULL
                  AND url IS NOT NULL
                  AND score IS NOT NULL AND score >= 1
                  AND (dead IS NULL OR dead = false)
                ORDER BY time, id
                LIMIT 10
            """
            
            df = pd.read_sql_query(query, conn)
            print(f"📥 Loaded {len(df)} sample records")
            
            # Test preprocessing
            processed_df = preprocessor.preprocess_batch(df)
            print(f"📤 Processed to {len(processed_df)} records")
            
            # Show new columns
            original_cols = set(df.columns)
            new_cols = set(processed_df.columns) - original_cols
            print(f"✨ Added features: {sorted(new_cols)}")
            
            # Show sample processed data
            print("\n📊 Sample processed data:")
            for i, row in processed_df.head(2).iterrows():
                print(f"  Record {i+1}:")
                print(f"    Original title: {row['title'][:50]}...")
                print(f"    Score: {row['score']} -> Log score: {row.get('log_score', 'N/A'):.2f}")
                print(f"    Features: length={row.get('title_length', 'N/A')}, words={row.get('title_word_count', 'N/A')}")
                print(f"    Time features: hour={row.get('hour_of_day', 'N/A')}, weekend={row.get('is_weekend', 'N/A')}")
                print()
                
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Test preprocessing functionality")
    parser.add_argument('--connection-only', action='store_true',
                       help='Test source database connection only')
    
    args = parser.parse_args()
    
    print("🔬 Preprocessing Test Suite")
    print("=" * 50)
    
    # Test source connection
    if not test_source_connection():
        print("❌ Source connection failed - cannot proceed")
        sys.exit(1)
    
    if not args.connection_only:
        # Test preprocessing logic
        if not test_preprocessing_dry_run():
            print("❌ Preprocessing test failed")
            sys.exit(1)
        
        print("✅ All tests passed! The preprocessor is ready to use.")
        print("\n💡 To run actual preprocessing:")
        print("   1. Set up a target database")
        print("   2. Run: python batch_preprocessor.py --limit 1000")
    else:
        print("✅ Source connection test passed!")
