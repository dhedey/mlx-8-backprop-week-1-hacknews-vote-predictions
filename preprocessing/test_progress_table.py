#!/usr/bin/env python3
"""
Test progress table creation for the batch preprocessor.
"""

import os
import sys
from batch_preprocessor import BatchPreprocessor, DatabaseConfig


def test_progress_table_creation():
    """Test just the progress table creation without full processing."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    # Use target database config
    target_config = DatabaseConfig.from_env("TARGET")
    if not target_config.database:
        target_config = DatabaseConfig.from_env()
    
    # Use source config too for completeness
    source_config = DatabaseConfig.from_env("SOURCE")
    
    print("🧪 Testing Progress Table Creation")
    print("=" * 50)
    print(f"Target DB: {target_config.host}:{target_config.port}/{target_config.database}")
    
    try:
        # Initialize preprocessor - this should create the progress table
        preprocessor = BatchPreprocessor(
            source_config=source_config,
            target_config=target_config,
            source_table='hacker_news.items',
            target_table='processed.hackernews_items',
            batch_size=100,
            create_target_db=True  # This should create the database if needed
        )
        
        print(f"✅ Progress table status: {'Available' if preprocessor.progress_table_available else 'Disabled'}")
        print(f"✅ Progress table name: {preprocessor.progress_table_name}")
        
        # Test getting resume cursor (should be None for new table)
        resume_cursor = preprocessor._get_resume_cursor()
        print(f"📍 Resume cursor: {resume_cursor}")
        
        # Clean up
        preprocessor.source_engine.dispose()
        preprocessor.target_engine.dispose()
        
        print("✅ Progress table test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Progress table test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_progress_table_creation()
    sys.exit(0 if success else 1)
