import torch
from torch.utils.data import Dataset, DataLoader, random_split
import psycopg2
import pandas as pd
from typing import Tuple, Optional
import os
from sklearn.model_selection import train_test_split


class PostgreSQLDataset(Dataset):
    """
    PyTorch Dataset for loading data from PostgreSQL database.
    """
    
    def __init__(self, connection_params: dict, table_name: str, 
                 columns: list, filter_condition: str = None):
        """
        Initialize the PostgreSQL Dataset.
        
        Args:
            connection_params (dict): Database connection parameters
            table_name (str): Name of the table to query
            columns (list): List of column names to select
            filter_condition (str): WHERE clause condition
        """
        self.connection_params = connection_params
        self.table_name = table_name
        self.columns = columns
        self.filter_condition = filter_condition
        
        # Load data from database
        self.data = self._load_data()
        
    def _load_data(self) -> pd.DataFrame:
        """
        Load data from PostgreSQL database.
        
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            # Connect to PostgreSQL
            conn = psycopg2.connect(**self.connection_params)
            
            # Build SQL query
            columns_str = ', '.join(self.columns)
            if self.filter_condition:
                query = f"""
                    SELECT {columns_str}
                    FROM {self.table_name}
                    WHERE {self.filter_condition}
                    ORDER BY id
                """
            else:
                query = f"""
                    SELECT {columns_str}
                    FROM {self.table_name}
                    ORDER BY id
                """
            
            print(f"Executing query: {query}")
            
            # Load data using pandas
            df = pd.read_sql_query(query, conn)
            
            # Close connection
            conn.close()
            
            print(f"Loaded {len(df)} rows from {self.table_name}")
            return df
            
        except Exception as e:
            print(f"Error loading data from database: {e}")
            raise
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Sample data as dictionary
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = self.data.iloc[idx].to_dict()
        return sample


class HackerNewsDataLoader:
    """
    DataLoader wrapper for HackerNews data with train/test split functionality.
    """
    
    def __init__(self, connection_params: dict, table_name: str = 'items_by_month',
                 columns: list = None, filter_condition: str = None,
                 train_split: float = 0.8, batch_size: int = 32, 
                 shuffle: bool = True, random_state: int = 42):
        """
        Initialize the HackerNews DataLoader.
        
        Args:
            connection_params (dict): Database connection parameters
            table_name (str): Name of the table to query
            columns (list): List of column names to select
            filter_condition (str): WHERE clause condition
            train_split (float): Fraction of data to use for training (default: 0.8)
            batch_size (int): Batch size for DataLoader
            shuffle (bool): Whether to shuffle the data
            random_state (int): Random seed for reproducibility
        """
        self.connection_params = connection_params
        self.table_name = table_name
        self.columns = columns or ['id', 'title', 'score']
        self.filter_condition = filter_condition or """type = 'story'
        AND title IS NOT NULL
        AND url IS NOT NULL
        AND score IS NOT NULL AND score >= 1
        AND (dead IS NULL OR dead = false)"""
        self.train_split = train_split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        
        # Set random seed
        torch.manual_seed(random_state)
        
        # Load dataset
        self.dataset = PostgreSQLDataset(
            connection_params=self.connection_params,
            table_name=self.table_name,
            columns=self.columns,
            filter_condition=self.filter_condition
        )
        
        # Create train/test split
        self.train_dataset, self.test_dataset = self._create_train_test_split()
        
    def _create_train_test_split(self) -> Tuple[Dataset, Dataset]:
        """
        Create train/test split of the dataset.
        
        Returns:
            Tuple[Dataset, Dataset]: Train and test datasets
        """
        dataset_size = len(self.dataset)
        train_size = int(self.train_split * dataset_size)
        test_size = dataset_size - train_size
        
        train_dataset, test_dataset = random_split(
            self.dataset, 
            [train_size, test_size],
            generator=torch.Generator().manual_seed(self.random_state)
        )
        
        print(f"Dataset split - Train: {train_size}, Test: {test_size}")
        return train_dataset, test_dataset
    
    def get_train_loader(self) -> DataLoader:
        """
        Get the training DataLoader.
        
        Returns:
            DataLoader: Training data loader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=0  # Set to 0 for database connections
        )
    
    def get_test_loader(self) -> DataLoader:
        """
        Get the test DataLoader.
        
        Returns:
            DataLoader: Test data loader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0  # Set to 0 for database connections
        )
    
    def get_data_info(self) -> dict:
        """
        Get information about the loaded dataset.
        
        Returns:
            dict: Dataset information
        """
        return {
            'total_samples': len(self.dataset),
            'train_samples': len(self.train_dataset),
            'test_samples': len(self.test_dataset),
            'columns': self.columns,
            'table_name': self.table_name,
            'filter_condition': self.filter_condition
        }


def create_connection_params(host: str = 'localhost', port: int = 5432, 
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
    # Try to get connection params from environment variables if not provided
    return {
        'host': host or os.getenv('POSTGRES_HOST', 'localhost'),
        'port': port or int(os.getenv('POSTGRES_PORT', 5432)),
        'database': database or os.getenv('POSTGRES_DB'),
        'user': user or os.getenv('POSTGRES_USER'),
        'password': password or os.getenv('POSTGRES_PASSWORD')
    }


# Example usage
if __name__ == "__main__":
    # Example connection parameters (replace with your actual database credentials)
    connection_params = create_connection_params(
        host='localhost',
        port=5432,
        database='your_database',
        user='your_username',
        password='your_password'
    )
    
    # Create data loader
    data_loader = HackerNewsDataLoader(
        connection_params=connection_params,
        table_name='items_by_month',
        columns=['id', 'title', 'score'],
        filter_condition=None,  # Uses default HackerNews filter
        train_split=0.8,
        batch_size=32,
        shuffle=True
    )
    
    # Get train and test loaders
    train_loader = data_loader.get_train_loader()
    test_loader = data_loader.get_test_loader()
    
    # Print dataset info
    info = data_loader.get_data_info()
    print("Dataset Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Example: iterate through first batch of training data
    print("\nFirst training batch:")
    for batch in train_loader:
        print(f"Batch keys: {batch.keys()}")
        if 'id' in batch:
            print(f"IDs: {batch['id'][:5]}...")  # Show first 5 IDs
        if 'title' in batch:
            print(f"Titles: {batch['title'][:2]}...")  # Show first 2 titles
        if 'score' in batch:
            print(f"Scores: {batch['score'][:5]}...")  # Show first 5 scores
        break
