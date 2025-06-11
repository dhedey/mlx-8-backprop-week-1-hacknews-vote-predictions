"""
Example script showing how to use the PostgreSQL DataLoader with a neural network model.
This integrates the database data loader with the existing model architecture.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import HackerNewsDataLoader, create_connection_params
import os
import math
from typing import Dict, Any


class HackerNewsNet(nn.Module):
    """
    Neural network for HackerNews score prediction.
    """
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128, 
                 hidden_dim: int = 256, output_dim: int = 1):
        super(HackerNewsNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
    def forward(self, x):
        # x should be tokenized text indices
        x = self.embedding(x)
        x = torch.mean(x, dim=1)  # Simple averaging of embeddings
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def simple_tokenizer(text: str, vocab_size: int = 10000) -> torch.Tensor:
    """
    Simple tokenizer that converts text to tensor of indices.
    In a real implementation, you'd use a proper tokenizer.
    """
    # This is a placeholder - replace with actual tokenization
    words = text.lower().split()[:20]  # Limit to 20 words
    # Simple hash-based tokenization (not recommended for production)
    indices = [hash(word) % vocab_size for word in words]
    
    # Pad or truncate to fixed length
    seq_length = 20
    if len(indices) < seq_length:
        indices.extend([0] * (seq_length - len(indices)))
    else:
        indices = indices[:seq_length]
        
    return torch.tensor(indices, dtype=torch.long)


def collate_fn(batch):
    """
    Custom collate function to handle text data from database.
    """
    ids = []
    titles = []
    scores = []
    
    for sample in batch:
        ids.append(sample['id'])
        titles.append(sample['title'] if sample['title'] else "")
        scores.append(sample['score'])
    
    # Tokenize titles
    tokenized_titles = torch.stack([simple_tokenizer(title) for title in titles])
    
    # Convert scores to log scores
    log_scores = torch.tensor([math.log(score) for score in scores], dtype=torch.float32)
    
    return {
        'ids': torch.tensor(ids),
        'titles': tokenized_titles,
        'title_texts': titles,  # Keep original text for reference
        'scores': torch.tensor(scores, dtype=torch.float32),
        'log_scores': log_scores
    }


def train_model_epoch(model, data_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    total_batches = 0
    
    for batch_idx, batch in enumerate(data_loader):
        # Move data to device
        titles = batch['titles'].to(device)
        log_scores = batch['log_scores'].to(device).unsqueeze(1)  # Add dimension for regression
        
        optimizer.zero_grad()
        outputs = model(titles)
        loss = criterion(outputs, log_scores)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        total_batches += 1
        
        if batch_idx % 10 == 9:
            avg_loss = running_loss / 10
            print(f'Batch: {batch_idx + 1}, Loss: {avg_loss:.4f}')
            running_loss = 0.0
    
    return running_loss / total_batches if total_batches > 0 else 0.0


def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate the model on test data.
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0
    predictions = []
    actual_log_scores = []
    
    with torch.no_grad():
        for batch in data_loader:
            titles = batch['titles'].to(device)
            log_scores = batch['log_scores'].to(device).unsqueeze(1)
            
            outputs = model(titles)
            loss = criterion(outputs, log_scores)
            total_loss += loss.item()
            total_batches += 1
            
            # Store predictions for analysis
            predictions.extend(outputs.cpu().numpy().flatten())
            actual_log_scores.extend(log_scores.cpu().numpy().flatten())
    
    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    
    # Calculate some statistics
    predictions = torch.tensor(predictions)
    actual_log_scores = torch.tensor(actual_log_scores)
    
    # Convert back to actual scores for interpretability
    predicted_scores = torch.exp(predictions)
    actual_scores = torch.exp(actual_log_scores)
    
    print(f'Test Loss (MSE on log scores): {avg_loss:.4f}')
    print(f'Mean predicted log score: {predictions.mean():.4f} (score: {predicted_scores.mean():.2f})')
    print(f'Mean actual log score: {actual_log_scores.mean():.4f} (score: {actual_scores.mean():.2f})')
    
    return avg_loss


def main():
    """
    Main function demonstrating the complete workflow.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Database connection parameters
    # Replace these with your actual database credentials
    connection_params = create_connection_params(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        database=os.getenv('POSTGRES_DB', 'your_database'),
        user=os.getenv('POSTGRES_USER', 'your_username'),
        password=os.getenv('POSTGRES_PASSWORD', 'your_password')
    )
    
    try:
        # Create data loader
        print("Creating data loader...")
        data_loader = HackerNewsDataLoader(
            connection_params=connection_params,
            table_name='items_by_month',
            columns=['id', 'title', 'score'],
            filter_condition=None,  # Uses default filter for HackerNews stories
            train_split=0.8,
            batch_size=16,
            shuffle=True
        )
        
        # Get train and test loaders with custom collate function
        train_loader = torch.utils.data.DataLoader(
            data_loader.train_dataset,
            batch_size=16,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        test_loader = torch.utils.data.DataLoader(
            data_loader.test_dataset,
            batch_size=16,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # Print dataset info
        info = data_loader.get_data_info()
        print("\nDataset Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Initialize model
        model = HackerNewsNet(vocab_size=10000, embedding_dim=128).to(device)
        criterion = nn.MSELoss()  # MSE loss for predicting log scores
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        print('\nStarting training...')
        num_epochs = 3
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            train_loss = train_model_epoch(model, train_loader, criterion, optimizer, device)
            test_loss = evaluate_model(model, test_loader, criterion, device)
            
        # Save the trained model
        model_path = 'hackernews_model.pth'
        torch.save(model.state_dict(), model_path)
        print(f'\nModel saved to {model_path}')
        
        # Example: Show some sample data
        print("\nSample data from first batch:")
        for batch in train_loader:
            print(f"Batch size: {len(batch['ids'])}")
            print(f"Sample IDs: {batch['ids'][:3].tolist()}")
            print(f"Sample titles: {batch['title_texts'][:2]}")
            print(f"Sample scores: {batch['scores'][:3].tolist()}")
            print(f"Sample log scores: {batch['log_scores'][:3].tolist()}")
            print(f"Tokenized shape: {batch['titles'].shape}")
            break
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to:")
        print("1. Set up your PostgreSQL database connection parameters")
        print("2. Ensure the 'items_by_month' table exists")
        print("3. Install required dependencies with uv")


if __name__ == '__main__':
    main()
