"""
Example script showing how to use the PostgreSQL DataLoader with a neural network model.
This integrates the database data loader with the existing model architecture.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import re
import pandas as pd
from pprint import pprint
import os
from dotenv import load_dotenv
import datasets
from torch.utils.data import DataLoader
import datetime
import argparse

class ModelConfiguration:
    def __init__(self, domain_counts_df, domain_min_count, author_counts_df, author_min_count, device, vocabulary, vocabulary_embeddings):
        self.domain_map = {
            x: i for i, x in enumerate(domain_counts_df[domain_counts_df["count"] >= domain_min_count]["domain"])
        }
        self.author_map = {
            x: i for i, x in enumerate(author_counts_df[author_counts_df["count"] >= author_min_count]["author"])
        }
        self.author_counts_df = author_counts_df
        self.vocabulary_embeddings = torch.cat([
            vocabulary_embeddings,
            torch.zeros((1, vocabulary_embeddings.shape[1]), dtype=torch.float32)  # Placeholder for unknown words
        ]).to(device)
        assert vocabulary_embeddings.shape[0] == len(vocabulary)
        self.title_vocab_map = { word: i for i, word in enumerate(vocabulary) }
        self.title_embedding_size = vocabulary_embeddings.shape[1]
        self.title_token_length = 20
        self.device = device

    def _tokenize_title(self, title):
        filtered_title_words = re.sub(r'[^a-z0-9 ]', '', title.lower()).split()
        token_placeholder = len(self.title_vocab_map) # Placeholder
        tokens = [token_placeholder] * self.title_token_length  # Initialize with placeholder tokens

        i = 0
        for word in filtered_title_words:
            if word in self.title_vocab_map:
                tokens[i] = self.title_vocab_map[word]
                i += 1
                if i >= self.title_token_length:
                    break

        return tokens

    def _map_author_id(self, author):
        if author in self.author_map:
            return self.author_map[author]
        else:
            return len(self.author_map) # Unknown

    def _map_domain_id(self, domain):
        if domain in self.domain_map:
            return self.domain_map[domain]
        else:
            return len(self.domain_map)  # Unknown

    def dimensions(self):
        return {
            "title_vocab_size": len(self.title_vocab_map) + 1, # Include placeholder
            "title_embedding_size": self.title_embedding_size,
            "title_token_length": self.title_token_length, # Anything more than this will be truncated, or padded
            "authors": len(self.author_map) + 1, # Include unknown
            "author_embedding_size": 16,
            "domains": len(self.domain_map) + 1,
            "domain_embedding_size": 16,
            "time_features": 5, # year, day of week cos/sin, hour of day cos/sin
            "hidden_dim_1_size": 256,
            "hidden_dim_2_size": 512,
        }

    def prepare_batch(self, batch):
        # The streaming loader already returns tensors for numeric data and lists for text
        ids = [id for id in batch['id']]
        scores = [score for score in batch['score']]

        tokenized_titles = [self._tokenize_title(title) for title in batch['title']]

        author_ids = [self._map_author_id(author) for author in batch['author']]
        domain_ids = [self._map_domain_id(domain) for domain in batch['domain']]

        timestamps = [datetime.datetime.fromtimestamp(time, datetime.UTC) for time in batch['time']]
        year = [date.year - 2000 for date in timestamps]
        day_of_week = [date.weekday() for date in timestamps]
        import math
        day_of_week_cos = [math.cos(2 * math.pi * date.weekday()/7) for date in timestamps]
        day_of_week_sin = [math.sin(2 * math.pi * date.weekday()/7) for date in timestamps]
        hour_of_day_cos = [math.cos(2 * math.pi * date.hour / 24) for date in timestamps]
        hour_of_day_sin = [math.sin(2 * math.pi * date.hour / 24) for date in timestamps]

        device = self.device

        return {
            'ids': torch.tensor(ids, dtype=torch.long).to(device),
            'log_scores': torch.log(torch.tensor([score for score in batch['score']], dtype=torch.float32) + 1).to(device),
            'features': {
                'tokenized_titles': torch.tensor(tokenized_titles, dtype=torch.int).to(device),
                'author_id': torch.tensor(author_ids, dtype=torch.int).to(device),
                'domain_id': torch.tensor(domain_ids, dtype=torch.int).to(device),
                'time': torch.stack(
                    [
                        torch.tensor(year, dtype=torch.int).to(device),
                        torch.tensor(day_of_week_cos, dtype=torch.int).to(device),
                        torch.tensor(day_of_week_sin, dtype=torch.int).to(device),
                        torch.tensor(hour_of_day_cos, dtype=torch.int).to(device),
                        torch.tensor(hour_of_day_sin, dtype=torch.int).to(device),
                    ],
                    dim=1,
                ).to(device),
            }
        }

class HackerNewsNet(nn.Module):
    """
    Neural network for HackerNews score prediction.
    """
    
    def __init__(self, configuration: ModelConfiguration):
        super(HackerNewsNet, self).__init__()

        dimensions = configuration.dimensions()

        self.title_embedding = nn.Embedding(
            dimensions["title_vocab_size"],
            dimensions["title_embedding_size"],
            _weight=configuration.vocabulary_embeddings,
            # _freeze=True,
        )
        self.author_embedding = nn.Embedding(dimensions["authors"], dimensions["author_embedding_size"])
        self.domain_embedding = nn.Embedding(dimensions["domains"], dimensions["domain_embedding_size"])

        input_feature_length = dimensions["title_embedding_size"] + dimensions["author_embedding_size"] + dimensions["domain_embedding_size"] + dimensions["time_features"]
        hidden_dim_1_size = dimensions["hidden_dim_1_size"]
        hidden_dim_2_size = dimensions["hidden_dim_2_size"]

        self.device = configuration.device
        
        # Layers
        self.fc1 = nn.Linear(input_feature_length, hidden_dim_1_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim_1_size, hidden_dim_2_size)
        self.fc3 = nn.Linear(hidden_dim_2_size, 1)
        
    def forward(self, features):
        embedded_title_tokens = self.title_embedding(features['tokenized_titles']) # (batch, title_length, embedding_dim)

        # We now create features which should be of dimension (batch, feature_length)
        title_features = torch.mean(embedded_title_tokens, dim=1)
        time_features = features['time']
        domain_features = self.domain_embedding(features['domain_id'])
        author_features = self.author_embedding(features['author_id'])

        x = torch.cat( # Concatenate along the feature dimension
            [
                title_features,
                domain_features,
                author_features,
                time_features,
            ],
            dim=1,
        )
        
        # TODO: Improve model architecture!!
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return torch.squeeze(x, dim=1)  # The final dimension is size 1 - flatten it / remove it


def train_model_epoch(model, data_loader, criterion, optimizer, preparer):
    """
    Train the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    print_running_loss = 0.0
    print_every = 100

    total_batches = len(data_loader)

    for batch_idx, raw_batch in enumerate(data_loader):
        # Process the batch
        batch = preparer.prepare_batch(raw_batch)
        
        optimizer.zero_grad()
        outputs = model(batch['features'])
        loss = criterion(outputs, batch['log_scores'])
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        print_running_loss += loss.item()

        batch_num = batch_idx + 1
        if batch_num % print_every == 0:
            print(f'Batch: {batch_num} of {total_batches}, Recent loss: {print_running_loss / print_every:.4f}')
            print_running_loss = 0.0
    
    return running_loss / total_batches if total_batches > 0 else 0.0


def evaluate_model(model, data_loader, criterion, preparer):
    """
    Evaluate the model on test data.
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0
    predictions = []
    actual_log_scores = []

    print("Evaluating on test data...")

    with torch.no_grad():
        for raw_batch in data_loader:
            # Process the batch
            batch = preparer.prepare_batch(raw_batch)
            log_scores = batch['log_scores']
            
            outputs = model(batch['features'])
            loss = criterion(outputs, batch['log_scores'])
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
    predicted_scores = torch.exp(predictions) - 1
    actual_scores = torch.exp(actual_log_scores) - 1
    
    print(f'Test Loss (MSE on log scores): {avg_loss:.4f}')
    print(f'Mean predicted log score: {predictions.mean():.4f} (score: {predicted_scores.mean():.2f})')
    print(f'Mean actual log score: {actual_log_scores.mean():.4f} (score: {actual_scores.mean():.2f})')
    
    return avg_loss

def main():
    """
    Main function demonstrating the complete workflow.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train HackerNews score prediction model')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training and evaluation (default: 128)')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs (default: 3)')
    args = parser.parse_args()
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Batch size: {args.batch_size}')
    print(f'Epochs: {args.epochs}')

    print("Loading dataset...")
    datasets.config.IN_MEMORY_MAX_SIZE = 8 * 1024 * 1024 # 8GB

    # Load the filtered dataset
    folder = os.path.dirname(__file__)
    dataset = datasets.load_from_disk(folder + "/filtered_dataset")
    print(f"Dataset loaded, size: {len(dataset)}")

    (train_dataset, test_dataset) = torch.utils.data.random_split(
        dataset,
        [0.9, 0.1],
        torch.Generator().manual_seed(42)
    )
    print(f"Dataset loaded and split into train: {len(train_dataset)} and test: {len(test_dataset)}")

    # Get train and test loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    word_vectors = torch.load(folder + '/word_vectors.pt')

    configuration = ModelConfiguration(
        domain_counts_df=pd.read_csv(folder + '/domain_counts.csv'),
        domain_min_count=4,
        author_counts_df=pd.read_csv(folder + '/author_counts.csv'),
        author_min_count=3,
        vocabulary=word_vectors["vocabulary"],
        vocabulary_embeddings=word_vectors["embeddings"],
        device=device,
    )

    print("Configuration:")
    pprint(configuration.dimensions())

    # print("\nA small sample batch:")
    # sample_loader = DataLoader(train_dataset, batch_size=2)
    # for raw_batch in sample_loader:
    #     pprint(raw_batch)
    #     batch = configuration.prepare_batch(raw_batch)
    #     pprint(batch)
    #     break

    # Initialize model
    model = HackerNewsNet(configuration).to(device)
    criterion = nn.MSELoss()  # MSE loss for predicting log scores
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    print('\nStarting training...')
    num_epochs = args.epochs

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        train_loss = train_model_epoch(model, train_loader, criterion, optimizer, configuration)
        test_loss = evaluate_model(model, test_loader, criterion, configuration)

        model_path = folder + '/hackernews_model.pth'
        torch.save(model.state_dict(), model_path)
        print(f'\nModel saved to {model_path}')

    # Save the trained model
    model_path = folder + '/hackernews_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f'\nModel saved to {model_path}')


if __name__ == '__main__':
    main()
