"""
Configuration file for database connections and model parameters.
"""

import os
from typing import Dict, Any


class Config:
    """Configuration class for the HackerNews prediction project."""
    
    # Database configuration
    DB_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    DB_PORT = int(os.getenv('POSTGRES_PORT', 5432))
    DB_NAME = os.getenv('POSTGRES_DB', 'hackernews')
    DB_USER = os.getenv('POSTGRES_USER', 'postgres')
    DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', '')
    
    # Model configuration
    VOCAB_SIZE = 10000
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 5
    TRAIN_SPLIT = 0.8
    RANDOM_SEED = 42
    
    # Data configuration
    TABLE_NAME = 'items_by_month'
    COLUMNS = ['id', 'title', 'score']
    FILTER_CONDITION = """type = 'story'
        AND title IS NOT NULL
        AND url IS NOT NULL
        AND score IS NOT NULL AND score >= 1
        AND (dead IS NULL OR dead = false)"""
    MAX_SEQUENCE_LENGTH = 50
    
    @classmethod
    def get_db_params(cls) -> Dict[str, Any]:
        """Get database connection parameters."""
        return {
            'host': cls.DB_HOST,
            'port': cls.DB_PORT,
            'database': cls.DB_NAME,
            'user': cls.DB_USER,
            'password': cls.DB_PASSWORD
        }
    
    @classmethod
    def get_model_params(cls) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'vocab_size': cls.VOCAB_SIZE,
            'embedding_dim': cls.EMBEDDING_DIM,
            'hidden_dim': cls.HIDDEN_DIM,
            'output_dim': cls.OUTPUT_DIM
        }
    
    @classmethod
    def get_training_params(cls) -> Dict[str, Any]:
        """Get training parameters."""
        return {
            'batch_size': cls.BATCH_SIZE,
            'learning_rate': cls.LEARNING_RATE,
            'num_epochs': cls.NUM_EPOCHS,
            'train_split': cls.TRAIN_SPLIT,
            'random_seed': cls.RANDOM_SEED
        }
