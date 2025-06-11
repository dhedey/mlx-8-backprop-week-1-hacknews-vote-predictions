#!/usr/bin/env python3
"""
Example usage of the batch preprocessor.
This script demonstrates how to use the BatchPreprocessor class programmatically.
"""

import os
import sys
from batch_preprocessor import BatchPreprocessor, DatabaseConfig


def example_preprocessing():
    """Example of running the batch preprocessor programmatically."""
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("Warning: python-dotenv not installed")
    
    # Database configurations
    # Using same database for source and target in this example
    db_config = DatabaseConfig.from_env()
    
    print(f"Connecting to: {db_config.host}:{db_config.port}/{db_config.database}")
    
    # Initialize preprocessor with auto-resume enabled
    preprocessor = BatchPreprocessor(
        source_config=db_config,
        target_config=db_config,  # Same database
        source_table='hacker_news.items',  # Source table
        target_table='processed.hackernews_features',  # Target table
        batch_size=500,  # Smaller batches for example
        create_target_db=True  # Auto-create database if needed
    )
    
    # Define columns to select
    columns = [
        'id',
        'title', 
        'score',
        'time',
        'type',
        'url',
        'dead',
        'by'  # author field
    ]
    
    # Define filter condition (same as your SQL query)
    where_clause = """
        type = 'story'
        AND title IS NOT NULL
        AND url IS NOT NULL
        AND score IS NOT NULL AND score >= 1
        AND (dead IS NULL OR dead = false)
    """
    
    # Add limit for demo (first 5000 records)
    demo_where_clause = where_clause + " AND id <= (SELECT MIN(id) + 5000 FROM hacker_news.items_by_month WHERE " + where_clause.strip() + ")"
    
    try:
        # Run preprocessing with auto-resume enabled
        print("🚀 Starting preprocessing pipeline...")
        print("📝 This will automatically resume if interrupted and restarted")
        
        preprocessor.run(
            columns=columns,
            where_clause=demo_where_clause,
            auto_resume=True  # Automatically resume from last processed position
        )
        
        print("✅ Preprocessing completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏸️  Processing interrupted! You can resume by running this script again.")
        print("The progress has been saved automatically.")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        sys.exit(1)


def test_connection():
    """Test database connection."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    db_config = DatabaseConfig.from_env()
    
    print("Testing database connection...")
    print(f"Host: {db_config.host}")
    print(f"Port: {db_config.port}")
    print(f"Database: {db_config.database}")
    print(f"User: {db_config.user}")
    
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(db_config.get_connection_url())
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.scalar()
            print(f"✅ Connection successful!")
            print(f"PostgreSQL version: {version}")
            
            # Test source table
            result = conn.execute(text("SELECT COUNT(*) FROM hacker_news.items_by_month LIMIT 1"))
            print(f"✅ Source table accessible")
            
        engine.dispose()
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
    
    return True


def show_progress_status():
    """Show current processing status from progress table."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    # Use target database config
    target_config = DatabaseConfig.from_env("TARGET")
    if not target_config.database:
        target_config = DatabaseConfig.from_env()
    
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(target_config.get_connection_url())
        
        with engine.connect() as conn:
            # First check if any progress tables exist
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'processed' 
                AND table_name LIKE '%_progress'
                ORDER BY table_name
            """))
            
            progress_tables = [row[0] for row in result.fetchall()]
            
            if not progress_tables:
                print("📋 No processing history found - no progress tables exist yet")
                print("💡 Run the preprocessor first to create progress tracking")
                return
            
            print("📊 Processing Status:")
            print("=" * 80)
            
            # Query each progress table
            for table_name in progress_tables:
                try:
                    result = conn.execute(text(f"""
                        SELECT source_table, target_table, status, total_processed,
                               last_processed_time, last_processed_id,
                               started_at, updated_at
                        FROM processed.{table_name}
                        ORDER BY updated_at DESC
                        LIMIT 5
                    """))
                    
                    rows = result.fetchall()
                    
                    if rows:
                        print(f"📋 Progress Table: {table_name}")
                        print("-" * 40)
                        
                        for row in rows:
                            status_icon = '🟢' if row[2] == 'completed' else '🟡' if row[2] == 'running' else '🔴'
                            
                            print(f"Source: {row[0]}")
                            print(f"Target: {row[1]}")
                            print(f"Status: {row[2]} {status_icon}")
                            print(f"Records Processed: {row[3]:,}")
                            
                            if row[4] and row[5]:  # last_processed_time and id
                                print(f"Last Cursor: ({row[4]}, {row[5]})")
                            
                            print(f"Started: {row[6]}")
                            print(f"Updated: {row[7]}")
                            print("─" * 30)
                        
                        print()
                        
                except Exception as table_error:
                    print(f"⚠️  Could not read progress table {table_name}: {table_error}")
            
        engine.dispose()
        
    except Exception as e:
        if "does not exist" in str(e) or "relation" in str(e):
            print("📋 No processing history found - progress tables don't exist yet")
            print("💡 Run the preprocessor first to create progress tracking")
        else:
            print(f"❌ Could not fetch progress status: {e}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Example batch preprocessor usage")
    parser.add_argument('--test-connection', action='store_true',
                       help='Test database connection only')
    parser.add_argument('--show-progress', action='store_true',
                       help='Show current processing progress/status')
    
    args = parser.parse_args()
    
    if args.test_connection:
        test_connection()
    elif args.show_progress:
        show_progress_status()
    else:
        example_preprocessing()
