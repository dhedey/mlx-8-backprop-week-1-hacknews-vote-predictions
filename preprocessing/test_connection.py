#!/usr/bin/env python3
"""
Quick connection test for the batch preprocessor.
"""

import os
import sys
from sqlalchemy import create_engine, text
from batch_preprocessor import DatabaseConfig


def test_connection():
    """Test database connections quickly."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    print("🔗 Testing Database Connections")
    print("=" * 50)
    
    # Test source connection
    source_config = DatabaseConfig.from_env("SOURCE")
    print(f"Source DB: {source_config.host}:{source_config.port}/{source_config.database}")
    
    try:
        source_engine = create_engine(source_config.get_connection_url(), pool_pre_ping=True)
        with source_engine.connect() as conn:
            # Quick test query
            result = conn.execute(text("SELECT COUNT(*) FROM hacker_news.items_by_month LIMIT 1"))
            print("✅ Source database connection successful")
        source_engine.dispose()
    except Exception as e:
        print(f"❌ Source database connection failed: {e}")
        return False
    
    # Test target connection
    target_config = DatabaseConfig.from_env("TARGET")
    if not target_config.database:
        target_config = DatabaseConfig.from_env()
    
    print(f"Target DB: {target_config.host}:{target_config.port}/{target_config.database}")
    
    try:
        target_engine = create_engine(target_config.get_connection_url(), pool_pre_ping=True)
        with target_engine.connect() as conn:
            # Quick test query
            result = conn.execute(text("SELECT 1"))
            print("✅ Target database connection successful")
        target_engine.dispose()
    except Exception as e:
        print(f"❌ Target database connection failed: {e}")
        return False
    
    print("✅ All database connections working!")
    return True


if __name__ == '__main__':
    success = test_connection()
    sys.exit(0 if success else 1)
