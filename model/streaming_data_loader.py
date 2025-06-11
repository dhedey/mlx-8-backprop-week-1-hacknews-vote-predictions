import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import urllib.parse
import os
from typing import Tuple, Optional, Iterator
from sqlalchemy import create_engine, text
from tqdm import tqdm


class StreamingPostgreSQLDataset(Dataset):
    """
    PyTorch Dataset that streams data from PostgreSQL database on-demand.
    Uses ID modulo 10 for deterministic train/test split.
    """
    
    def __init__(self, connection_params: dict, table_name: str, 
                 columns: list, filter_condition: str = None,
                 is_train: bool = True, train_split_mod: int = 8,
                 batch_size: int = 1000, order_by: str = 'time'):
        """
        Initialize the Streaming PostgreSQL Dataset.
        
        Args:
            connection_params (dict): Database connection parameters
            table_name (str): Name of the table to query
            columns (list): List of column names to select
            filter_condition (str): WHERE clause condition
            is_train (bool): Whether this is training data (True) or test data (False)
            train_split_mod (int): Use ID % 10 < train_split_mod for training (default: 8 = 80%)
            batch_size (int): Number of rows to fetch per database query
            order_by (str): Column to order the data by (default: 'time')
        """
        self.connection_params = connection_params
        self.table_name = table_name
        self.columns = columns
        self.filter_condition = filter_condition
        self.is_train = is_train
        self.train_split_mod = train_split_mod
        self.batch_size = batch_size
        self.order_by = order_by
        
        # Create SQLAlchemy engine
        password = urllib.parse.quote_plus(self.connection_params['password'])
        connection_url = (
            f"postgresql://{self.connection_params['user']}:{password}@"
            f"{self.connection_params['host']}:{self.connection_params['port']}/"
            f"{self.connection_params['database']}"
        )
        self.engine = create_engine(connection_url, pool_pre_ping=True)
        
        # Get dataset size and create split condition
        self._setup_split_condition()
        self._total_count = self._get_total_count()
        
        print(f"{'Training' if is_train else 'Test'} dataset: {self._total_count:,} samples")
        
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
    
    def _build_query(self, offset: int, limit: int) -> str:
        """Build SQL query with pagination."""
        columns_str = ', '.join(self.columns)
        
        conditions = []
        if self.filter_condition:
            conditions.append(self.filter_condition)
        conditions.append(self.split_condition)
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
            SELECT {columns_str}
            FROM {self.table_name}
            WHERE {where_clause}
            ORDER BY {self.order_by}
            LIMIT {limit} OFFSET {offset}
        """
        return query
    
    def _fetch_batch(self, offset: int, limit: int) -> pd.DataFrame:
        """Fetch a batch of data from the database."""
        query = self._build_query(offset, limit)
        
        with self.engine.connect() as conn:
            try:
                df = pd.read_sql_query(query, conn)
            except Exception as e:
                print("ERROR RUNNING QUERY:")
                print(query)
                raise e

        return df
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self._total_count
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a sample from the dataset by fetching from database.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Sample data as dictionary
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Fetch single row (inefficient for individual access, but works for batched access)
        df = self._fetch_batch(idx, 1)
        
        if len(df) == 0:
            raise IndexError(f"Index {idx} out of range")
        
        return df.iloc[0].to_dict()


class BatchedStreamingDataLoader:
    """
    Custom DataLoader that fetches data in batches from the database.
    More efficient than individual row access.
    """
    
    def __init__(self, dataset: StreamingPostgreSQLDataset, batch_size: int = 32, 
                 shuffle: bool = False):
        """
        Initialize the BatchedStreamingDataLoader.
        
        Args:
            dataset: The streaming dataset
            batch_size: Number of samples per batch for training
            shuffle: Whether to shuffle (not implemented for streaming)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._total_samples = len(dataset)
        
        if shuffle:
            print("Warning: Shuffle not implemented for streaming data loader. Data will be in time order.")
    
    def __iter__(self) -> Iterator[dict]:
        """Iterate through the dataset in batches."""
        offset = 0
        
        while offset < self._total_samples:
            # Fetch database batch (larger than training batch for efficiency)
            db_batch_size = max(self.dataset.batch_size, self.batch_size * 4)
            df_batch = self.dataset._fetch_batch(offset, db_batch_size)
            
            if len(df_batch) == 0:
                break
            
            # Yield training batches from the database batch
            for i in range(0, len(df_batch), self.batch_size):
                batch_df = df_batch.iloc[i:i + self.batch_size]
                
                if len(batch_df) == 0:
                    break
                
                # Convert to batch dictionary
                batch = {}
                for col in batch_df.columns:
                    if col in ['id', 'score']:
                        # Convert to list first, then to tensor to avoid SQLAlchemy issues
                        values = batch_df[col].astype(int).tolist()
                        batch[col] = torch.tensor(values, dtype=torch.long if col == 'id' else torch.float32)
                    else:
                        # Convert to list for text columns
                        batch[col] = batch_df[col].astype(str).tolist()
                
                # Debug: Print batch info for first batch
                if offset == 0 and i == 0:
                    print(f"First batch debug:")
                    print(f"  df_batch shape: {batch_df.shape}")
                    print(f"  df_batch columns: {list(batch_df.columns)}")
                    print(f"  batch keys: {list(batch.keys())}")
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            print(f"  {key}: tensor shape {value.shape}, dtype {value.dtype}")
                        else:
                            print(f"  {key}: list of length {len(value)}")
                
                yield batch
            
            offset += len(df_batch)
    
    def __len__(self) -> int:
        """Return the number of batches."""
        return (self._total_samples + self.batch_size - 1) // self.batch_size


class HackerNewsStreamingDataLoader:
    """
    Streaming DataLoader wrapper for HackerNews data with train/test split using ID modulo.
    """
    
    def __init__(self, connection_params: dict, table_name: str = None,
                 columns: list = None, filter_condition: str = None,
                 train_split: float = 0.8, batch_size: int = 32, 
                 shuffle: bool = True, db_batch_size: int = 1000, 
                 order_by: str = 'time'):
        """
        Initialize the HackerNews Streaming DataLoader.
        
        Args:
            connection_params (dict): Database connection parameters
            table_name (str): Name of the table to query
            columns (list): List of column names to select
            filter_condition (str): WHERE clause condition
            train_split (float): Fraction of data to use for training (default: 0.8)
            batch_size (int): Batch size for training/inference
            shuffle (bool): Whether to shuffle (warning: not implemented for streaming)
            db_batch_size (int): Number of rows to fetch per database query
            order_by (str): Column to order the data by (default: 'time')
        """
        self.connection_params = connection_params
        self.table_name = table_name or 'hacker_news.items_by_month'
        self.columns = columns or ['id', 'title', 'score']
        self.filter_condition = filter_condition or """type = 'story'
        AND title IS NOT NULL
        AND url IS NOT NULL
        AND score IS NOT NULL AND score >= 1
        AND (dead IS NULL OR dead = false)"""
        self.train_split = train_split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.db_batch_size = db_batch_size
        self.order_by = order_by
        
        # Convert train_split to modulo value (0.8 -> 8, 0.7 -> 7, etc.)
        self.train_split_mod = int(train_split * 10)
        
        print(f"Using ID % 10 split: Train (< {self.train_split_mod}), Test (>= {self.train_split_mod})")
        
        # Create train and test datasets
        self.train_dataset = StreamingPostgreSQLDataset(
            connection_params=self.connection_params,
            table_name=self.table_name,
            columns=self.columns,
            filter_condition=self.filter_condition,
            is_train=True,
            train_split_mod=self.train_split_mod,
            batch_size=self.db_batch_size,
            order_by=self.order_by
        )
        
        self.test_dataset = StreamingPostgreSQLDataset(
            connection_params=self.connection_params,
            table_name=self.table_name,
            columns=self.columns,
            filter_condition=self.filter_condition,
            is_train=False,
            train_split_mod=self.train_split_mod,
            batch_size=self.db_batch_size,
            order_by=self.order_by
        )
        
        print(f"Dataset split - Train: {len(self.train_dataset):,}, Test: {len(self.test_dataset):,}")
        
    def get_train_loader(self) -> BatchedStreamingDataLoader:
        """Get the training DataLoader."""
        return BatchedStreamingDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle
        )
    
    def get_test_loader(self) -> BatchedStreamingDataLoader:
        """Get the test DataLoader."""
        return BatchedStreamingDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
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
            'order_by': self.order_by
        }
    
    def close(self):
        """Close database connections."""
        if hasattr(self.train_dataset, 'engine'):
            self.train_dataset.engine.dispose()
        if hasattr(self.test_dataset, 'engine'):
            self.test_dataset.engine.dispose()


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


# Example usage
if __name__ == "__main__":
    # Create connection parameters from environment
    connection_params = create_connection_params()
    
    # Create streaming data loader
    data_loader = HackerNewsStreamingDataLoader(
        connection_params=connection_params,
        table_name='hacker_news.items_by_month',
        columns=['id', 'title', 'score'],
        filter_condition=None,  # Uses default HackerNews filter
        train_split=0.8,
        batch_size=16,
        shuffle=False,
        db_batch_size=1000
    )
    
    # Get train and test loaders
    train_loader = data_loader.get_train_loader()
    test_loader = data_loader.get_test_loader()
    
    # Print dataset info
    info = data_loader.get_data_info()
    print("\nDataset Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Example: iterate through first few batches
    print("\nFirst few training batches:")
    for i, batch in enumerate(train_loader):
        print(f"Batch {i+1}:")
        print(f"  Batch size: {len(batch['id'])}")
        print(f"  Sample IDs: {batch['id'][:3].tolist()}")
        print(f"  Sample titles: {batch['title'][:2]}")
        if 'score' in batch:
            print(f"  Sample scores: {batch['score'][:3].tolist()}")
        
        if i >= 2:  # Only show first 3 batches
            break
    
    # Clean up
    data_loader.close()
