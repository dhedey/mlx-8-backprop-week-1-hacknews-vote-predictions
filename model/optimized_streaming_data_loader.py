import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import urllib.parse
import os
import threading
import queue
import time
from typing import Tuple, Optional, Iterator, Dict, Any
from sqlalchemy import create_engine, text
from tqdm import tqdm


class OptimizedStreamingPostgreSQLDataset(Dataset):
    """
    High-performance PyTorch Dataset that streams data from PostgreSQL database.
    
    Key optimizations:
    1. Uses cursor-based pagination with (time, id) instead of slow OFFSET queries
    2. Background loading with buffer to preload batches ahead of time
    3. Uses ID modulo 10 for deterministic train/test split
    """
    
    def __init__(self, connection_params: dict, table_name: str, 
                 columns: list, filter_condition: str = None,
                 is_train: bool = True, train_split_mod: int = 8,
                 batch_size: int = 1000, order_by: str = 'time',
                 buffer_size: int = 5):
        """
        Initialize the Optimized Streaming PostgreSQL Dataset.
        
        Args:
            connection_params (dict): Database connection parameters
            table_name (str): Name of the table to query
            columns (list): List of column names to select
            filter_condition (str): WHERE clause condition
            is_train (bool): Whether this is training data (True) or test data (False)
            train_split_mod (int): Use ID % 10 < train_split_mod for training (default: 8 = 80%)
            batch_size (int): Number of rows to fetch per database query
            order_by (str): Column to order the data by (default: 'time')
            buffer_size (int): Number of batches to preload in background (default: 5)
        """
        self.connection_params = connection_params
        self.table_name = table_name
        self.columns = columns
        self.filter_condition = filter_condition
        self.is_train = is_train
        self.train_split_mod = train_split_mod
        self.batch_size = batch_size
        self.order_by = order_by
        self.buffer_size = buffer_size
        
        # Create SQLAlchemy engine
        password = urllib.parse.quote_plus(self.connection_params['password'])
        connection_url = (
            f"postgresql://{self.connection_params['user']}:{password}@"
            f"{self.connection_params['host']}:{self.connection_params['port']}/"
            f"{self.connection_params['database']}"
        )
        self.engine = create_engine(connection_url, pool_pre_ping=True)
        
        # Setup split condition and get dataset info
        self._setup_split_condition()
        self._total_count = self._get_total_count()
        self._cursor_info = self._get_cursor_bounds()
        
        # Background loading setup
        self._batch_queue = queue.Queue(maxsize=buffer_size)
        self._loading_thread = None
        self._stop_loading = threading.Event()
        self._loading_started = False
        
        print(f"{'Training' if is_train else 'Test'} dataset: {self._total_count:,} samples")
        print(f"Cursor range: {self._cursor_info['min_time']} to {self._cursor_info['max_time']}")
        
    def _setup_split_condition(self):
        """Setup the SQL condition for train/test split based on ID modulo."""
        if self.is_train:
            self.split_condition = f"MOD(id, 10) < {self.train_split_mod}"
        else:
            self.split_condition = f"MOD(id, 10) >= {self.train_split_mod}"
    
    def _get_total_count(self) -> int:
        """Get the total count of items for this split."""
        conditions = []
        if self.filter_condition:
            conditions.append(self.filter_condition)
        conditions.append(self.split_condition)
        
        where_clause = " AND ".join(conditions)
        
        count_query = f"""
            SELECT COUNT(*) as total_count
            FROM {self.table_name}
            WHERE {where_clause}
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(count_query))
            return result.scalar()
    
    def _get_cursor_bounds(self) -> Dict[str, Any]:
        """Get the min/max cursor values for pagination."""
        conditions = []
        if self.filter_condition:
            conditions.append(self.filter_condition)
        conditions.append(self.split_condition)
        
        where_clause = " AND ".join(conditions)
        
        bounds_query = f"""
            SELECT 
                MIN({self.order_by}) as min_time,
                MAX({self.order_by}) as max_time,
                MIN(id) as min_id,
                MAX(id) as max_id
            FROM {self.table_name}
            WHERE {where_clause}
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(bounds_query))
            row = result.fetchone()
            return {
                'min_time': row[0],
                'max_time': row[1],
                'min_id': row[2],
                'max_id': row[3]
            }
    
    def _build_cursor_query(self, cursor_time: Any = None, cursor_id: int = None, 
                           limit: int = None) -> str:
        """
        Build cursor-based query using (time, id) for efficient pagination.
        Much faster than OFFSET on large datasets.
        """
        columns_str = ', '.join(self.columns)
        
        conditions = []
        if self.filter_condition:
            conditions.append(self.filter_condition)
        conditions.append(self.split_condition)
        
        # Add cursor condition for pagination
        if cursor_time is not None and cursor_id is not None:
            cursor_condition = f"""
                ({self.order_by} > '{cursor_time}' 
                 OR ({self.order_by} = '{cursor_time}' AND id > {cursor_id}))
            """
            conditions.append(cursor_condition)
        
        where_clause = " AND ".join(conditions)
        limit_clause = f"LIMIT {limit}" if limit else ""
        
        query = f"""
            SELECT {columns_str}
            FROM {self.table_name}
            WHERE {where_clause}
            ORDER BY {self.order_by}, id
            {limit_clause}
        """
        return query
    
    def _fetch_batch_cursor(self, cursor_time: Any = None, cursor_id: int = None, 
                           limit: int = None) -> pd.DataFrame:
        """Fetch a batch using cursor-based pagination."""
        query = self._build_cursor_query(cursor_time, cursor_id, limit)
        
        with self.engine.connect() as conn:
            try:
                df = pd.read_sql_query(query, conn)
            except Exception as e:
                print("ERROR RUNNING CURSOR QUERY:")
                print(query)
                raise e
        
        return df
    
    def _background_loader(self):
        """Background thread function to preload batches."""
        cursor_time = None
        cursor_id = None
        
        try:
            while not self._stop_loading.is_set():
                # Fetch next batch from database
                df = self._fetch_batch_cursor(cursor_time, cursor_id, self.batch_size)
                
                if len(df) == 0:
                    # No more data, signal end
                    self._batch_queue.put(None)
                    break
                
                # Update cursor to last row
                last_row = df.iloc[-1]
                cursor_time = last_row[self.order_by]
                cursor_id = last_row['id']
                
                # Add batch to queue (this will block if queue is full)
                if not self._stop_loading.is_set():
                    self._batch_queue.put(df)
                
        except Exception as e:
            print(f"Background loader error: {e}")
            # Put error in queue to signal main thread
            self._batch_queue.put(e)
    
    def _start_background_loading(self):
        """Start background loading thread."""
        if not self._loading_started:
            self._loading_started = True
            self._loading_thread = threading.Thread(target=self._background_loader, daemon=True)
            self._loading_thread.start()
            print("Started background batch loading...")
    
    def get_batch_iterator(self) -> Iterator[pd.DataFrame]:
        """Get iterator over database batches using background loading."""
        self._start_background_loading()
        
        while True:
            try:
                # Get next batch from queue (blocks until available)
                batch = self._batch_queue.get(timeout=30.0)  # 30 second timeout
                
                if batch is None:
                    # End of data signal
                    break
                
                if isinstance(batch, Exception):
                    # Error from background thread
                    raise batch
                
                yield batch
                
            except queue.Empty:
                print("Warning: Timeout waiting for next batch from background loader")
                break
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self._total_count
    
    def close(self):
        """Clean up resources."""
        self._stop_loading.set()
        if self._loading_thread and self._loading_thread.is_alive():
            self._loading_thread.join(timeout=5.0)
        if hasattr(self, 'engine'):
            self.engine.dispose()


class OptimizedBatchedStreamingDataLoader:
    """
    High-performance DataLoader that uses background loading and cursor pagination.
    Eliminates database query bottlenecks during training.
    """
    
    def __init__(self, dataset: OptimizedStreamingPostgreSQLDataset, 
                 training_batch_size: int = 32, shuffle: bool = False):
        """
        Initialize the Optimized Batched Streaming DataLoader.
        
        Args:
            dataset: The optimized streaming dataset
            training_batch_size: Number of samples per training batch
            shuffle: Whether to shuffle (not implemented for streaming)
        """
        self.dataset = dataset
        self.training_batch_size = training_batch_size
        self.shuffle = shuffle
        self._total_samples = len(dataset)
        
        if shuffle:
            print("Warning: Shuffle not implemented for streaming data loader. Data will be in time order.")
    
    def __iter__(self) -> Iterator[Dict[str, list]]:
        """Iterate through the dataset in training batches using background-loaded DB batches."""
        total_yielded = 0
        
        # Use tqdm for progress tracking
        with tqdm(total=self._total_samples, desc="Loading batches") as pbar:
            
            for df_batch in self.dataset.get_batch_iterator():
                # Yield training batches from each database batch
                for i in range(0, len(df_batch), self.training_batch_size):
                    training_batch_df = df_batch.iloc[i:i + self.training_batch_size]
                    
                    if len(training_batch_df) == 0:
                        break
                    
                    # Convert to batch dictionary
                    batch = {}
                    for col in training_batch_df.columns:
                        batch[col] = training_batch_df[col].tolist()
                    
                    total_yielded += len(training_batch_df)
                    pbar.update(len(training_batch_df))
                    
                    yield batch
                    
                    if total_yielded >= self._total_samples:
                        break
    
    def __len__(self) -> int:
        """Return the number of training batches."""
        return (self._total_samples + self.training_batch_size - 1) // self.training_batch_size


class HackerNewsOptimizedDataLoader:
    """
    Optimized DataLoader wrapper for HackerNews data with high-performance streaming.
    
    Key features:
    - Background batch loading with buffer
    - Cursor-based pagination (no slow OFFSET queries)
    - Deterministic train/test split using ID modulo
    - Memory efficient streaming
    """
    
    def __init__(self, connection_params: dict, table_name: str,
                 columns: list, filter_condition: str = None,
                 train_split: float = 0.8, training_batch_size: int = 32, 
                 shuffle: bool = True, db_batch_size: int = 1000, 
                 order_by: str = 'time', buffer_size: int = 5):
        """
        Initialize the HackerNews Optimized DataLoader.
        
        Args:
            connection_params (dict): Database connection parameters
            table_name (str): Name of the table to query
            columns (list): List of column names to select
            filter_condition (str): WHERE clause condition
            train_split (float): Fraction of data to use for training (default: 0.8)
            training_batch_size (int): Batch size for training/inference
            shuffle (bool): Whether to shuffle (warning: not implemented for streaming)
            db_batch_size (int): Number of rows to fetch per database query
            order_by (str): Column to order the data by (default: 'time')
            buffer_size (int): Number of batches to preload (default: 5)
        """
        self.connection_params = connection_params
        self.table_name = table_name
        self.columns = columns
        self.filter_condition = filter_condition
        self.train_split = train_split
        self.training_batch_size = training_batch_size
        self.shuffle = shuffle
        self.db_batch_size = db_batch_size
        self.order_by = order_by
        self.buffer_size = buffer_size
        
        # Convert train_split to modulo value (0.8 -> 8, 0.7 -> 7, etc.)
        self.train_split_mod = int(train_split * 10)
        
        print(f"🚀 OPTIMIZED DATALOADER - Using background loading + cursor pagination")
        print(f"Using ID % 10 split: Train (< {self.train_split_mod}), Test (>= {self.train_split_mod})")
        print(f"Buffer size: {buffer_size} batches, DB batch size: {db_batch_size}")
        
        # Create train and test datasets
        self.train_dataset = OptimizedStreamingPostgreSQLDataset(
            connection_params=self.connection_params,
            table_name=self.table_name,
            columns=self.columns,
            filter_condition=self.filter_condition,
            is_train=True,
            train_split_mod=self.train_split_mod,
            batch_size=self.db_batch_size,
            order_by=self.order_by,
            buffer_size=self.buffer_size
        )
        
        self.test_dataset = OptimizedStreamingPostgreSQLDataset(
            connection_params=self.connection_params,
            table_name=self.table_name,
            columns=self.columns,
            filter_condition=self.filter_condition,
            is_train=False,
            train_split_mod=self.train_split_mod,
            batch_size=self.db_batch_size,
            order_by=self.order_by,
            buffer_size=self.buffer_size
        )
        
        print(f"Dataset split - Train: {len(self.train_dataset):,}, Test: {len(self.test_dataset):,}")
        
    def get_train_loader(self) -> OptimizedBatchedStreamingDataLoader:
        """Get the optimized training DataLoader."""
        return OptimizedBatchedStreamingDataLoader(
            self.train_dataset,
            training_batch_size=self.training_batch_size,
            shuffle=self.shuffle
        )
    
    def get_test_loader(self) -> OptimizedBatchedStreamingDataLoader:
        """Get the optimized test DataLoader."""
        return OptimizedBatchedStreamingDataLoader(
            self.test_dataset,
            training_batch_size=self.training_batch_size,
            shuffle=False
        )
    
    def get_data_info(self) -> dict:
        """Get information about the loaded dataset."""
        return {
            'total_samples': len(self.train_dataset) + len(self.test_dataset),
            'train_samples': len(self.train_dataset),
            'test_samples': len(self.test_dataset),
            'columns': self.columns,
            'table_name': self.table_name,
            'filter_condition': self.filter_condition,
            'train_split_mod': self.train_split_mod,
            'order_by': self.order_by,
            'db_batch_size': self.db_batch_size,
            'buffer_size': self.buffer_size
        }
    
    def close(self):
        """Close database connections and stop background threads."""
        print("Closing optimized data loader...")
        self.train_dataset.close()
        self.test_dataset.close()


def create_connection_params(host: str = None, port: int = None, 
                           database: str = None, user: str = None, 
                           password: str = None) -> dict:
    """
    Create database connection parameters.
    
    Args:
        host (str): Database host
        port (int): Database port
        database (str): Database name
        user (str): Database user
        password (str): Database password
        
    Returns:
        dict: Connection parameters
    """
    return {
        'host': host or os.getenv('POSTGRES_HOST', 'localhost'),
        'port': port or int(os.getenv('POSTGRES_PORT', 5432)),
        'database': database or os.getenv('POSTGRES_DB'),
        'user': user or os.getenv('POSTGRES_USER'),
        'password': password or os.getenv('POSTGRES_PASSWORD')
    }


# Example usage and testing
if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    
    print("Testing Optimized HackerNews DataLoader...")
    
    # Create connection parameters
    connection_params = create_connection_params()
    
    print(f"Connecting to: {connection_params['host']}:{connection_params['port']}/{connection_params['database']}")
    
    # Test the optimized data loader
    data_loader = HackerNewsOptimizedDataLoader(
        connection_params=connection_params,
        table_name='hacker_news.items_by_month',
        columns=['id', 'title', 'score'],
        filter_condition="""
            type = 'story'
            AND title IS NOT NULL
            AND url IS NOT NULL
            AND score IS NOT NULL AND score >= 1
            AND (dead IS NULL OR dead = false)
        """,
        train_split=0.8,
        training_batch_size=32,
        db_batch_size=1000,
        buffer_size=5
    )
    
    # Test train loader
    print("\n🔥 Testing optimized train loader (first 3 batches):")
    train_loader = data_loader.get_train_loader()
    
    start_time = time.time()
    batch_count = 0
    
    for batch in train_loader:
        batch_count += 1
        print(f"Batch {batch_count}: {len(batch['id'])} samples")
        print(f"  Sample titles: {batch['title'][:2]}")
        print(f"  Sample scores: {batch['score'][:2]}")
        
        if batch_count >= 3:  # Only test first 3 batches
            break
    
    elapsed = time.time() - start_time
    print(f"\n✅ Loaded {batch_count} batches in {elapsed:.2f} seconds")
    if batch_count > 0:
        print(f"   Average: {elapsed/batch_count:.2f} seconds per batch")
    
    # Clean up
    data_loader.close()
    print("✅ Test completed successfully!")
