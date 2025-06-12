#!/usr/bin/env python3
"""
Batch Preprocessor for HackerNews Data

Streams data from PostgreSQL source database, applies preprocessing transformations,
and writes results to a target database using efficient cursor-based pagination.

Features:
- Cursor-based pagination over (time, id) for efficient streaming
- Configurable batch processing
- Data validation and cleaning
- Text preprocessing (tokenization, normalization)
- Progress tracking with resumable processing
- Error handling and logging
"""

import os
import sys
import time
import logging
import argparse
from typing import Dict, Any, Optional, Iterator, List, Tuple
from datetime import datetime, timezone
import urllib.parse

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Float, DateTime, Boolean, create_engine
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import OperationalError, ProgrammingError
from tqdm import tqdm
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


class DatabaseConfig:
    """Database configuration management."""
    
    def __init__(self, host: str, port: int, database: str, user: str, password: str):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
    
    @classmethod
    def from_env(cls, prefix: str = ""):
        """Create config from environment variables."""
        prefix = prefix + "_" if prefix else ""
        return cls(
            host=os.getenv(f'{prefix}POSTGRES_HOST', 'localhost'),
            port=int(os.getenv(f'{prefix}POSTGRES_PORT', 5432)),
            database=os.getenv(f'{prefix}POSTGRES_DB'),
            user=os.getenv(f'{prefix}POSTGRES_USER'),
            password=os.getenv(f'{prefix}POSTGRES_PASSWORD')
        )
    
    def get_connection_url(self, include_db: bool = True) -> str:
        """Get SQLAlchemy connection URL."""
        password = urllib.parse.quote_plus(self.password)
        db_part = f"/{self.database}" if include_db and self.database else ""
        return (
            f"postgresql://{self.user}:{password}@"
            f"{self.host}:{self.port}{db_part}"
        )
    
    def create_database_if_not_exists(self):
        """Create database if it doesn't exist."""
        if not self.database:
            raise ValueError("Database name not specified")
        
        # Connect without specifying database
        engine = create_engine(self.get_connection_url(include_db=False))
        
        try:
            with engine.connect() as conn:
                # Check if database exists
                result = conn.execute(text(
                    "SELECT 1 FROM pg_database WHERE datname = :db_name"
                ), {"db_name": self.database})
                
                if not result.fetchone():
                    # Database doesn't exist, create it
                    conn.execute(text("COMMIT"))  # Close any transaction
                    conn.execute(text(f'CREATE DATABASE "{self.database}"'))
                    print(f"✅ Created database: {self.database}")
                else:
                    print(f"✅ Database already exists: {self.database}")
                    
        except Exception as e:
            print(f"⚠️  Could not create database {self.database}: {e}")
            print("Please ensure you have CREATE DATABASE privileges or create the database manually")
        finally:
            engine.dispose()


class CursorPaginator:
    """Handles cursor-based pagination over (time, id) columns."""
    
    def __init__(self, engine, table_name: str, columns: List[str], 
                 where_clause: str = None, order_by: str = "time",
                 batch_size: int = 1000, limit: int = None):
        self.engine = engine
        self.table_name = table_name
        self.columns = columns
        self.where_clause = where_clause
        self.order_by = order_by
        self.batch_size = batch_size
        self.limit = limit
        
        # Get cursor bounds
        self.cursor_bounds = self._get_cursor_bounds()
        self.total_count = self._get_total_count()
        
    def _get_cursor_bounds(self) -> Dict[str, Any]:
        """Get min/max cursor values for pagination."""
        try:
            conditions = [self.where_clause] if self.where_clause else []
            where_sql = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            
            bounds_query = f"""
                SELECT 
                    MIN({self.order_by}) as min_time,
                    MAX({self.order_by}) as max_time,
                    MIN(id) as min_id,
                    MAX(id) as max_id
                FROM {self.table_name}
                {where_sql}
            """
            
            print(f"📊 Getting cursor bounds from {self.table_name}...")
            with self.engine.connect() as conn:
                result = conn.execute(text(bounds_query))
                row = result.fetchone()
                bounds = {
                    'min_time': row[0],
                    'max_time': row[1],
                    'min_id': row[2],
                    'max_id': row[3]
                }
                print(f"✅ Cursor bounds: {bounds}")
                return bounds
        except Exception as e:
            print(f"❌ Error getting cursor bounds: {e}")
            raise
    
    def _get_total_count(self) -> int:
        """Get total count of records."""
        try:
            conditions = [self.where_clause] if self.where_clause else []
            where_sql = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            
            count_query = f"SELECT COUNT(*) FROM {self.table_name} {where_sql}"
            
            print(f"📊 Getting total count from {self.table_name}...")
            with self.engine.connect() as conn:
                result = conn.execute(text(count_query))
                count = result.scalar()
                print(f"✅ Total records: {count:,}")
                return count
        except Exception as e:
            print(f"❌ Error getting total count: {e}")
            raise
    
    def _build_cursor_query(self, cursor_time: Any = None, cursor_id: int = None) -> str:
        """Build cursor-based query for efficient pagination."""
        columns_str = ', '.join(self.columns)
        
        conditions = []
        if self.where_clause:
            conditions.append(self.where_clause)
        
        # Add cursor condition for pagination
        if cursor_time is not None and cursor_id is not None:
            cursor_condition = f"""
                ({self.order_by} > '{cursor_time}' 
                 OR ({self.order_by} = '{cursor_time}' AND id > {cursor_id}))
            """
            conditions.append(cursor_condition)
        
        where_sql = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        
        query = f"""
            SELECT {columns_str}
            FROM {self.table_name}
            {where_sql}
            ORDER BY {self.order_by}, id
            LIMIT {self.batch_size}
        """
        return query
    
    def iterate_batches(self, resume_cursor: Tuple[Any, int] = None) -> Iterator[pd.DataFrame]:
        """Iterate through data in batches using cursor pagination."""
        cursor_time, cursor_id = resume_cursor if resume_cursor else (None, None)
        processed_count = 0
        
        # Use limit if specified, otherwise use total_count
        total_to_process = min(self.limit, self.total_count) if self.limit else self.total_count
        
        with tqdm(total=total_to_process, desc="Processing batches") as pbar:
            while True:
                # Stop if we've reached the limit
                if self.limit and processed_count >= self.limit:
                    break
                    
                query = self._build_cursor_query(cursor_time, cursor_id)
                
                with self.engine.connect() as conn:
                    print(f"📄 Executing query: {query}")
                    df = pd.read_sql_query(query, conn)
                    print(f"... query complete!")
                
                if len(df) == 0:
                    break
                
                # If limit specified, trim the dataframe
                if self.limit:
                    remaining = self.limit - processed_count
                    if len(df) > remaining:
                        df = df.head(remaining)
                
                # Update cursor to last row
                last_row = df.iloc[-1]
                cursor_time = last_row[self.order_by]
                cursor_id = last_row['id']
                
                processed_count += len(df)
                pbar.update(len(df))
                
                yield df, (cursor_time, cursor_id)
                
                if len(df) < self.batch_size:
                    break


class DataPreprocessor:
    """Handles data preprocessing transformations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def preprocess_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing transformations to a batch.
        
        Args:
            df: Input batch DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        processed_df = df.copy()
        
        # 1. Data validation and cleaning
        processed_df = self._validate_and_clean(processed_df)
        
        # 2. Text preprocessing
        if 'title' in processed_df.columns:
            processed_df = self._preprocess_text(processed_df)
        
        # 3. Feature engineering
        processed_df = self._engineer_features(processed_df)
        
        # 4. Score transformations
        if 'score' in processed_df.columns:
            processed_df = self._transform_scores(processed_df)
        
        return processed_df
    
    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean data."""
        # Remove rows with invalid scores
        if 'score' in df.columns:
            df = df[df['score'] >= 1]
            df = df[df['score'].notna()]
        
        # Clean title text
        if 'title' in df.columns:
            df = df[df['title'].notna()]
            df = df[df['title'].str.len() > 0]
            # Remove excessive whitespace
            df['title'] = df['title'].str.strip()
            df['title'] = df['title'].str.replace(r'\s+', ' ', regex=True)
        
        return df
    
    def _preprocess_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess text fields."""
        if 'title' in df.columns:
            # Normalize case
            df['title_normalized'] = df['title'].str.lower()
            
            # Calculate text features
            df['title_length'] = df['title'].str.len()
            df['title_word_count'] = df['title'].str.split().str.len()
            
            # Extract question marks, exclamation points
            df['has_question'] = df['title'].str.contains(r'\?', regex=True).astype(int)
            df['has_exclamation'] = df['title'].str.contains(r'!', regex=True).astype(int)
            
            # Extract numbers
            df['has_numbers'] = df['title'].str.contains(r'\d', regex=True).astype(int)
            
            # Common tech keywords
            tech_keywords = ['AI', 'ML', 'Python', 'JavaScript', 'React', 'API', 'GitHub', 'Open Source']
            for keyword in tech_keywords:
                df[f'has_{keyword.lower().replace(" ", "_")}'] = (
                    df['title'].str.contains(keyword, case=False, regex=False).astype(int)
                )
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features."""
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            
            # Time-based features
            df['hour_of_day'] = df['time'].dt.hour
            df['day_of_week'] = df['time'].dt.dayofweek  # 0=Monday, 6=Sunday
            df['day_of_month'] = df['time'].dt.day
            df['month'] = df['time'].dt.month
            df['year'] = df['time'].dt.year
            
            # Weekend indicator
            df['is_weekend'] = (df['day_of_week'].isin([5, 6])).astype(int)
            
            # Business hours
            df['is_business_hours'] = (
                (df['hour_of_day'] >= 9) & (df['hour_of_day'] <= 17) & (~df['is_weekend'].astype(bool))
            ).astype(int)
        
        return df
    
    def _transform_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform score values."""
        if 'score' in df.columns:
            # Log transform
            df['log_score'] = np.log(df['score'])
            
            # Score categories
            df['score_category'] = pd.cut(
                df['score'], 
                bins=[0, 5, 15, 50, 150, float('inf')], 
                labels=['low', 'medium', 'high', 'very_high', 'viral']
            )
            
            # Z-score normalization (within batch)
            df['score_zscore'] = (df['score'] - df['score'].mean()) / df['score'].std()
        
        return df


class BatchPreprocessor:
    """Main batch preprocessing pipeline with resumable processing."""
    
    def __init__(self, source_config: DatabaseConfig, target_config: DatabaseConfig,
                 source_table: str, target_table: str, batch_size: int = 1000,
                 create_target_db: bool = True):
        self.source_config = source_config
        self.target_config = target_config
        self.batch_size = batch_size
        self.create_target_db = create_target_db
        
        # Setup logging first
        self.logger = self._setup_logging()
        
        # Create target database if needed
        if self.create_target_db:
            self.target_config.create_database_if_not_exists()
        
        # Initialize engines
        self.source_engine = create_engine(source_config.get_connection_url(), pool_pre_ping=True)
        self.target_engine = create_engine(target_config.get_connection_url(), pool_pre_ping=True)
        
        # Initialize preprocessor
        self.preprocessor = DataPreprocessor()
        
        # Extract schemas
        if '.' in source_table:
            schema, table = source_table.split('.', 1)

            self.source_schema = schema
            self.source_table = table
            self.full_source_table = f"{self.source_schema}.{self.source_table}"
        else:
            self.source_schema = None
            self.source_table = source_table
            self.full_source_table = f"{self.source_table}"

        if '.' in target_table:
            schema, table = target_table.split('.', 1)

            self.target_schema = schema
            self.target_table = table
            self.full_target_table = f"{self.target_schema}.{self.target_table}"
        else:
            self.target_schema = None
            self.target_table = target_table
            self.full_target_table = target_table

        # Progress tracking table name
        self.progress_table_name = f"{self.target_table}_progress"
        self.full_progress_table_name = f"{self.target_schema}.{self.progress_table_name}"
        
        # Create and verify progress table immediately
        self.progress_table_available = False
        try:
            self._create_progress_table()
            self.progress_table_available = True
            self.logger.info("✅ Progress tracking initialized successfully")
        except Exception as e:
            self.logger.warning(f"⚠️  Progress tracking disabled due to error: {e}")
            self.logger.warning("Processing will continue but won't be resumable")
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('preprocessing.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)
    
    def create_target_table(self, sample_df: pd.DataFrame):
        """Create target table based on sample data structure."""
        metadata = MetaData()
        
        # Define table columns based on DataFrame
        columns = [
            Column('id', Integer, primary_key=True),
            Column('processed_at', DateTime, default=lambda: datetime.now(timezone.utc)),
        ]
        
        # Add columns based on DataFrame
        for col, dtype in sample_df.dtypes.items():
            if col == 'id':
                continue
                
            if pd.api.types.is_integer_dtype(dtype):
                columns.append(Column(col, Integer))
            elif pd.api.types.is_float_dtype(dtype):
                columns.append(Column(col, Float))
            elif pd.api.types.is_bool_dtype(dtype):
                columns.append(Column(col, Boolean))
            elif pd.api.types.is_datetime64_dtype(dtype):
                columns.append(Column(col, DateTime))
            else:
                columns.append(Column(col, String))
        
        table = Table(self.target_table, metadata, *columns, schema=self.target_schema)
        metadata.create_all(self.target_engine)

        if self.target_schema is not None:
            self.logger.info(f"Created target table: {self.target_table} under schema {self.target_schema}")
        else:
            self.logger.info(f"Created target table: {self.target_table} under default schema")
        return table
    
    def process_and_save_batch(self, df: pd.DataFrame, target_table: Table):
        """Process a batch and save to target database."""
        try:
            # Preprocess the batch
            processed_df = self.preprocessor.preprocess_batch(df)
            
            if len(processed_df) == 0:
                self.logger.warning("Batch became empty after preprocessing")
                return 0
            
            # Add processing timestamp
            processed_df['processed_at'] = datetime.now(timezone.utc)
            
            # Insert into target database using upsert
            with self.target_engine.connect() as conn:
                for _, row in processed_df.iterrows():
                    stmt = insert(target_table).values(**row.to_dict())
                    # On conflict, update all columns except id
                    update_dict = {col.name: stmt.excluded[col.name] 
                                 for col in target_table.columns if col.name != 'id'}
                    stmt = stmt.on_conflict_do_update(
                        index_elements=['id'],
                        set_=update_dict
                    )
                    conn.execute(stmt)
                conn.commit()
            
            self.logger.info(f"Processed and saved batch: {len(processed_df)} rows")
            return len(processed_df)
            
        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")
            raise
    
    def _create_progress_table(self):
        """Create progress tracking table and ensure it exists."""
        # First, ensure the schema exists for the progress table
        if self.target_schema is not None:
            schema_name = self.target_schema
            try:
                with self.target_engine.connect() as conn:
                    conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"'))
                    conn.commit()
                    self.logger.info(f"✅ Ensured schema exists for progress table: {schema_name}")
            except Exception as e:
                self.logger.warning(f"Could not create schema {schema_name}: {e}")
                raise
        
        # Create the progress table
        metadata = MetaData()

        progress_table = Table(
            self.progress_table_name,
            metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('source_table', String, nullable=False),
            Column('target_table', String, nullable=False),
            Column('last_processed_time', DateTime, nullable=True),
            Column('last_processed_id', Integer, nullable=True),
            Column('total_processed', Integer, default=0),
            Column('started_at', DateTime, default=lambda: datetime.now(timezone.utc)),
            Column('updated_at', DateTime, default=lambda: datetime.now(timezone.utc),
                   onupdate=lambda: datetime.now(timezone.utc)),
            Column('status', String, default='running'),  # running, completed, failed
            schema=self.target_schema
        )

        try:
            # Create the progress table
            metadata.create_all(self.target_engine)
            self.logger.info(f"✅ Progress tracking table created: {self.progress_table_name} under schema {self.target_schema}")
            
            # Verify table exists and is accessible
            verify_query = f'SELECT COUNT(*) FROM "{self.full_progress_table_name}"'
            
            with self.target_engine.connect() as conn:
                result = conn.execute(text(verify_query))
                count = result.scalar()
                self.logger.info(f"✅ Progress table verified - contains {count} existing records")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create/verify progress table: {e}")
            self.logger.error("Progress tracking will not be available - resumability disabled")
            raise
        
        return progress_table
    
    def _get_resume_cursor(self) -> Optional[Tuple[Any, int]]:
        """Get the last processed cursor from progress table."""
        if not self.progress_table_available:
            return None
            
        try:
            with self.target_engine.connect() as conn:
                result = conn.execute(text(f"""
                    SELECT last_processed_time, last_processed_id, total_processed
                    FROM {self.full_progress_table_name}
                    WHERE source_table = :source_table 
                    AND target_table = :target_table
                    AND status = 'running'
                    ORDER BY updated_at DESC
                    LIMIT 1
                """), {
                    "source_table": self.source_table,
                    "target_table": self.target_table
                })
                
                row = result.fetchone()
                if row and row[0] and row[1]:
                    self.logger.info(f"Resuming from cursor: ({row[0]}, {row[1]}) - {row[2]:,} records processed")
                    return (row[0], row[1])
                    
        except Exception as e:
            self.logger.warning(f"Could not get resume cursor: {e}")
        
        return None
    
    def _update_progress(self, cursor_time: Any, cursor_id: int, processed_count: int):
        """Update progress in tracking table."""
        if not self.progress_table_available:
            return
            
        try:
            # Convert numpy types to Python native types
            cursor_id = int(cursor_id)
            processed_count = int(processed_count)
            
            with self.target_engine.connect() as conn:
                # Try to update existing record
                result = conn.execute(text(f"""
                    UPDATE {self.full_progress_table_name}
                    SET last_processed_time = :cursor_time,
                        last_processed_id = :cursor_id,
                        total_processed = total_processed + :processed_count,
                        updated_at = :updated_at
                    WHERE source_table = :source_table 
                    AND target_table = :target_table
                    AND status = 'running'
                """), {
                    "cursor_time": cursor_time,
                    "cursor_id": cursor_id,
                    "processed_count": processed_count,
                    "updated_at": datetime.now(timezone.utc),
                    "source_table": self.source_table,
                    "target_table": self.target_table
                })
                
                # If no rows updated, insert new record
                if result.rowcount == 0:
                    conn.execute(text(f"""
                        INSERT INTO {self.full_progress_table_name} 
                        (source_table, target_table, last_processed_time, last_processed_id, 
                         total_processed, started_at, updated_at, status)
                        VALUES (:source_table, :target_table, :cursor_time, :cursor_id,
                                :processed_count, :started_at, :updated_at, 'running')
                    """), {
                        "source_table": self.source_table,
                        "target_table": self.target_table,
                        "cursor_time": cursor_time,
                        "cursor_id": cursor_id,
                        "processed_count": processed_count,
                        "started_at": datetime.now(timezone.utc),
                        "updated_at": datetime.now(timezone.utc)
                    })
                
                conn.commit()
                
        except Exception as e:
            self.logger.warning(f"Could not update progress: {e}")
    
    def _mark_completed(self, total_processed: int):
        """Mark processing as completed."""
        if not self.progress_table_available:
            return
            
        try:
            with self.target_engine.connect() as conn:
                conn.execute(text(f"""
                    UPDATE {self.full_progress_table_name}
                    SET status = 'completed',
                        total_processed = :total_processed,
                        updated_at = :updated_at
                    WHERE source_table = :source_table 
                    AND target_table = :target_table
                    AND status = 'running'
                """), {
                    "total_processed": total_processed,
                    "updated_at": datetime.now(timezone.utc),
                    "source_table": self.source_table,
                    "target_table": self.target_table
                })
                conn.commit()
                
        except Exception as e:
            self.logger.warning(f"Could not mark as completed: {e}")
    
    def _create_target_schema_if_not_exists(self):
        """Create target schema if it doesn't exist."""
        if self.target_schema is not None:
            schema_name = self.target_schema
            try:
                with self.target_engine.connect() as conn:
                    conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"'))
                    conn.commit()
                    self.logger.info(f"Ensured schema exists: {schema_name}")
            except Exception as e:
                self.logger.warning(f"Could not create schema {schema_name}: {e}")
    
    def run(self, columns: List[str], where_clause: str = None, 
            resume_cursor: Tuple[Any, int] = None, auto_resume: bool = True,
            limit: int = None):
        """
        Run the complete preprocessing pipeline with automatic resumability.
        
        Args:
            columns: List of columns to select from source
            where_clause: WHERE condition for source query
            resume_cursor: Resume from specific cursor position (overrides auto_resume)
            auto_resume: Automatically resume from last processed position
            limit: Limit processing to N records (for testing)
        """
        self.logger.info("🚀 Starting batch preprocessing pipeline")
        self.logger.info(f"Source: {self.source_config.database}.{self.full_source_table}")
        self.logger.info(f"Target: {self.target_config.database}.{self.full_target_table}")
        
        # Progress tracking status
        if self.progress_table_available:
            self.logger.info("✅ Progress tracking enabled - processing is resumable")
        else:
            self.logger.warning("⚠️  Progress tracking disabled - processing not resumable")
        
        # Create target schema if needed
        self._create_target_schema_if_not_exists()
        
        # Determine resume cursor
        if resume_cursor is None and auto_resume:
            resume_cursor = self._get_resume_cursor()
            if resume_cursor:
                self.logger.info("✅ Auto-resuming from previous session")
            else:
                self.logger.info("🆕 Starting fresh processing session")
        elif resume_cursor:
            self.logger.info(f"📍 Resuming from manual cursor: {resume_cursor}")
        elif not auto_resume:
            self.logger.info("🔄 Auto-resume disabled - starting from beginning")
        
        # Initialize paginator
        paginator = CursorPaginator(
            engine=self.source_engine,
            table_name=self.full_source_table,
            columns=columns,
            where_clause=where_clause,
            batch_size=self.batch_size,
            limit=limit
        )
        
        self.logger.info(f"Total records to process: {paginator.total_count:,}")
        
        # Initialize processing variables
        target_table = None
        total_processed = 0
        
        try:
            for batch_num, (df, cursor) in enumerate(paginator.iterate_batches(resume_cursor), 1):
                if target_table is None:
                    # Create target table based on first processed batch
                    sample_processed = self.preprocessor.preprocess_batch(df.head(10))
                    target_table = self.create_target_table(sample_processed)
                
                # Process and save batch
                processed_count = self.process_and_save_batch(df, target_table)
                total_processed += processed_count
                
                # Update progress tracking
                cursor_time, cursor_id = cursor
                self._update_progress(cursor_time, cursor_id, processed_count)
                
                # Log progress
                self.logger.info(
                    f"Batch {batch_num}: Processed {processed_count} rows "
                    f"(Total: {total_processed:,}) - Cursor: ({cursor_time}, {cursor_id})"
                )
                
                # Checkpoint logging
                if batch_num % 10 == 0:
                    self.logger.info(f"💾 Checkpoint: Processed {total_processed:,} records")
        
        except KeyboardInterrupt:
            self.logger.info("⏸️  Processing interrupted by user. Progress saved for resuming.")
            raise
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            if self.progress_table_available:
                try:
                    # Try to update status to failed
                    with self.target_engine.connect() as conn:
                        conn.execute(text(f"""
                            UPDATE {self.full_progress_table_name}
                            SET status = 'failed', updated_at = :updated_at
                            WHERE source_table = :source_table AND target_table = :target_table
                        """), {
                            "updated_at": datetime.now(timezone.utc),
                            "source_table": self.source_table,
                            "target_table": self.target_table
                        })
                        conn.commit()
                except Exception as progress_error:
                    self.logger.warning(f"Could not update progress status to failed: {progress_error}")
            raise
        finally:
            # Cleanup connections
            self.source_engine.dispose()
            self.target_engine.dispose()
        
        # Mark as completed
        self._mark_completed(total_processed)
        self.logger.info(f"✅ Pipeline completed successfully! Processed {total_processed:,} records")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Resumable batch preprocessor for HackerNews data")
    parser.add_argument('--source-table', default='hacker_news.items', 
                       help='Source table name (default: hacker_news.items)')
    parser.add_argument('--target-table', default='processed.hackernews_items',
                       help='Target table name (default: processed.hackernews_items)')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for processing (default: 1000)')
    parser.add_argument('--resume-time', help='Resume from specific time cursor (overrides auto-resume)')
    parser.add_argument('--resume-id', type=int, help='Resume from specific ID cursor')
    parser.add_argument('--no-auto-resume', action='store_true',
                       help='Disable automatic resuming from progress table')
    parser.add_argument('--no-create-db', action='store_true',
                       help='Skip automatic database creation')
    parser.add_argument('--limit', type=int, help='Limit processing to N records (for testing)')
    
    args = parser.parse_args()
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("Warning: python-dotenv not installed, using system environment variables")
    
    # Database configurations
    source_config = DatabaseConfig.from_env("SOURCE")  # SOURCE_POSTGRES_HOST, etc.
    target_config = DatabaseConfig.from_env("TARGET")  # TARGET_POSTGRES_HOST, etc.
    
    # If no separate target config, use same as source
    if not target_config.database:
        target_config = DatabaseConfig.from_env()  # Use default POSTGRES_* vars
    
    print(f"🔗 Source: {source_config.host}:{source_config.port}/{source_config.database}")
    print(f"🎯 Target: {target_config.host}:{target_config.port}/{target_config.database}")
    
    # Define source columns
    columns = [
        'id',
        'title', 
        'score',
        'time',
        'type',
        'url',
        'dead'
    ]
    
    # Define filter condition (matching your SQL query)
    where_clause = """
        type = 'story'
        AND title IS NOT NULL
        AND url IS NOT NULL
        AND score IS NOT NULL AND score >= 1
        AND (dead IS NULL OR dead = false)
    """
    
    # Add limit for testing
    if args.limit:
        # Instead of a complex subquery, we'll let the paginator handle the limit
        print(f"⚠️  LIMITED to {args.limit} records for testing")
        # We'll modify the paginator to respect the limit
    
    # Determine resume cursor
    resume_cursor = None
    if args.resume_time and args.resume_id:
        resume_cursor = (args.resume_time, args.resume_id)
        print(f"📍 Manual resume cursor: {resume_cursor}")
    
    # Initialize preprocessor
    try:
        preprocessor = BatchPreprocessor(
            source_config=source_config,
            target_config=target_config,
            source_table=args.source_table,
            target_table=args.target_table,
            batch_size=args.batch_size,
            create_target_db=not args.no_create_db
        )
        
        # Run preprocessing pipeline
        preprocessor.run(
            columns=columns,
            where_clause=where_clause,
            resume_cursor=resume_cursor,
            auto_resume=not args.no_auto_resume,
            limit=args.limit
        )
        
        print("🎉 Processing completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏸️  Processing interrupted. You can resume later by running the same command.")
        print("The progress has been saved and will be automatically resumed next time.")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your database connection parameters in .env file")
        print("2. Ensure the source table exists and is accessible")
        print("3. Verify you have CREATE privileges on the target database")
        print("4. Check the preprocessing.log file for detailed error information")
        sys.exit(1)


if __name__ == '__main__':
    main()
