"""
Example script showing how to use the PostgreSQL DataLoader with a neural network model.
This integrates the database data loader with the existing model architecture.
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import math
import re
import pandas as pd
from pprint import pprint
import os
from dotenv import load_dotenv
import datasets
from torch.utils.data import DataLoader
import datetime
import argparse
import wandb
from dataclasses import dataclass, field

@dataclass
class ModelHyperparameters:
    batch_size: int = 128
    epochs: int = 5
    learning_rate: float = 0.001
    # Model structure
    dropout: float = 0.2
    hidden_dimensions: list[int] = field(default_factory=lambda: [256, 128, 64])
    freeze_embeddings: bool = False
    include_batch_norms: bool = False
    domain_embedding_size: int = 16
    author_embedding_size: int = 16

    def to_dict(self):
        return vars(self)

@dataclass
class FeatureParameters:
    title_token_ids: int
    title_embedding_size: int
    title_token_length: int
    author_token_ids: int
    domain_token_ids: int
    time_features: int

    def to_dict(self):
        return vars(self)

def create_id_map(df, min_count, column):
    return {
        x: i for i, x in enumerate(df[df["count"] >= min_count][column])
    }

class FeaturePreparer:
    def __init__(self, domain_map, author_map, vocabulary, vocabulary_embeddings, device, title_token_length=20):
        self.domain_map = domain_map
        self.author_map = author_map
        self.title_vocab_map = { word: i for i, word in enumerate(vocabulary) }
        self.initial_vocabulary_embeddings = torch.cat([
            vocabulary_embeddings,
            torch.zeros((1, vocabulary_embeddings.shape[1]), dtype=torch.float32)  # Placeholder padding word
        ]).to(device)
        assert vocabulary_embeddings.shape[0] == len(vocabulary)
        self.title_token_length = title_token_length
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

    def feature_parameters(self) -> FeatureParameters:
        return FeatureParameters(
            title_token_ids=len(self.title_vocab_map) + 1, # Include placeholder
            title_embedding_size=self.initial_vocabulary_embeddings.shape[1],
            title_token_length=self.title_token_length, # Anything more than this will be truncated, or padded
            author_token_ids=len(self.author_map) + 1, # Include unknown
            domain_token_ids=len(self.domain_map) + 1, # Include unknown
            time_features=5, # year, day of week cos/sin, hour of day cos/sin
        )

    def prepare_batch(self, batch):
        # The streaming loader already returns tensors for numeric data and lists for text
        ids = [id for id in batch['id']]
        scores = [score for score in batch['score']]

        tokenized_titles = [self._tokenize_title(title) for title in batch['title']]

        author_ids = [self._map_author_id(author) for author in batch['author']]
        domain_ids = [self._map_domain_id(domain) for domain in batch['domain']]

        timestamps = [datetime.datetime.fromtimestamp(time, datetime.UTC) for time in batch['time']]
        year = [date.year - 2000 for date in timestamps]
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
    
    def __init__(self, feature_preparer: FeaturePreparer, hyper_parameters: ModelHyperparameters):
        super(HackerNewsNet, self).__init__()

        feature_parameters = feature_preparer.feature_parameters()

        self.title_embedding = nn.Embedding(
            feature_parameters.title_token_ids,
            feature_parameters.title_embedding_size,
            _weight=feature_preparer.initial_vocabulary_embeddings,
            _freeze=hyper_parameters.freeze_embeddings,
        )
        self.author_embedding = nn.Embedding(feature_parameters.author_token_ids, hyper_parameters.author_embedding_size)
        self.domain_embedding = nn.Embedding(feature_parameters.domain_token_ids, hyper_parameters.domain_embedding_size)

        input_feature_length = feature_parameters.title_embedding_size + hyper_parameters.author_embedding_size + hyper_parameters.domain_embedding_size + feature_parameters.time_features

        self.device = feature_preparer.device
        
        # Layers
        input_layer_sizes = [input_feature_length] + hyper_parameters.hidden_dimensions
        output_layer_sizes = hyper_parameters.hidden_dimensions + [1]
        self.layers = nn.Sequential(OrderedDict([
            (f'layer {i + 1}', nn.Sequential(OrderedDict([
                ('linear', nn.Linear(input_size, output_size)),
                ('relu', nn.ReLU()),
                ('batch_norm', nn.BatchNorm1d(output_size) if hyper_parameters.include_batch_norms else nn.Identity()),
                ('dropoiut', nn.Dropout(p=hyper_parameters.dropout)),
            ])))
            for i, (input_size, output_size) in enumerate(zip(input_layer_sizes, output_layer_sizes))
        ]))
        
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

        x = self.layers(x)

        return torch.squeeze(x, dim=1)  # The final dimension is size 1 - flatten it / remove it


def train_model_epoch(model, data_loader, criterion, optimizer, preparer, epoch=None):
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

        # Log batch-level metrics to wandb
        if wandb.run is not None:
            wandb.log({
                "batch_loss": loss.item(),
                "batch": batch_idx + (epoch - 1) * total_batches if epoch else batch_idx
            })

        batch_num = batch_idx + 1
        if batch_num % print_every == 0:
            print(f'Batch: {batch_num} of {total_batches}, Recent loss: {print_running_loss / print_every:.4f}')
            print_running_loss = 0.0
    
    avg_loss = running_loss / total_batches if total_batches > 0 else 0.0
    
    # Log epoch-level training metrics to wandb
    if wandb.run is not None and epoch is not None:
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_loss,
        })
    
    return avg_loss


def evaluate_model(model, data_loader, criterion, preparer):
    """
    Evaluate the model on test data.
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0
    predicted_log_scores = []
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
            predicted_log_scores.extend(outputs.cpu().numpy().flatten())
            actual_log_scores.extend(log_scores.cpu().numpy().flatten())
    
    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    
    # Calculate some statistics
    predicted_log_scores = torch.tensor(predicted_log_scores)
    actual_log_scores = torch.tensor(actual_log_scores)
    
    # Convert back to actual scores for interpretability
    predicted_raw_scores = torch.exp(predicted_log_scores) - 1
    actual_raw_scores = torch.exp(actual_log_scores) - 1
    
    print(f'Test Loss (MSE on log scores): {avg_loss:.4f}')
    print(f'Mean predicted log score: {predicted_log_scores.mean():.4f} (std: {predicted_log_scores.std():.4f})')
    print(f'Mean actual    log score: {actual_log_scores.mean():.4f} (std: {actual_log_scores.std():.4f})')
    print(f'Mean predicted raw score: {predicted_raw_scores.mean():.2f} (std: {predicted_raw_scores.std():.4f})')
    print(f'Mean actual    raw score: {actual_raw_scores.mean():.4f} (std: {actual_raw_scores.std():.4f})')
    
    # Log test metrics to wandb
    if wandb.run is not None:
        wandb.log({
            "test_loss": avg_loss,
            "predicted_log_score_mean": predicted_log_scores.mean().item(),
            "predicted_log_score_std": predicted_log_scores.std().item(),
            "actual_log_score_mean": actual_log_scores.mean().item(),
            "actual_log_score_std": actual_log_scores.std().item(),
            "predicted_raw_score_mean": predicted_raw_scores.mean().item(),
            "predicted_raw_score_std": predicted_raw_scores.std().item(),
            "actual_raw_score_mean": actual_raw_scores.mean().item(),
            "actual_raw_score_std": actual_raw_scores.std().item(),
        })
    
    return avg_loss

def train_model(hyper_parameters: ModelHyperparameters, continue_training = False):
    """
    Train the model with given settings.
    
    Args:
        hyper_parameters: ModelRunSettings instance with hyperparameters
    
    Returns:
        dict: Training results with final test loss and other metrics
    """
    
    if not isinstance(hyper_parameters, ModelHyperparameters):
        raise TypeError("hyperparameters must be an instance of ModelHyperparameters")
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print("Hyperparameters:")
    pprint(hyper_parameters.to_dict())

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
    train_loader = DataLoader(train_dataset, batch_size=hyper_parameters.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=hyper_parameters.batch_size)

    word_vectors = torch.load(folder + '/word_vectors.pt')

    feature_preparer = FeaturePreparer(
        domain_map=create_id_map(
            df=pd.read_csv(folder + '/domain_counts.csv'),
            min_count=4,
            column="domain"
        ),
        author_map=create_id_map(
            df=pd.read_csv(folder + '/author_counts.csv'),
            min_count=3,
            column="author"
        ),
        vocabulary=word_vectors["vocabulary"],
        vocabulary_embeddings=word_vectors["embeddings"],
        device=device,
    )

    print("Feature parameters:")
    pprint(feature_preparer.feature_parameters().to_dict())

    model_path = folder + '/hackernews_model.pth'

    # Initialize model
    model = HackerNewsNet(feature_preparer, hyper_parameters).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hyper_parameters.learning_rate)

    if continue_training:
        print(f"Loading model and optimizer state from {model_path}...")
        try:
            loaded_data = torch.load(model_path)
            optimizer.load_state_dict(loaded_data['optimizer'])
            model.load_state_dict(loaded_data["model"])
            start_epoch = loaded_data['epoch']
        except (FileNotFoundError, KeyError) as e:
            print(f"Could not load model: {e}. Starting from scratch.")
            start_epoch = 1
    else:
        start_epoch = 1

    criterion = nn.MSELoss()  # MSE loss for predicting log scores

    # Training loop
    print('\nStarting training...')
    
    best_test_loss = float('inf')
    training_results = {
        'final_train_loss': 0.0,
        'final_test_loss': 0.0,
        'best_test_loss': float('inf'),
        'epochs_completed': 0
    }

    for epoch in range(start_epoch, hyper_parameters.epochs + 1):
        print(f'\nEpoch {epoch}/{hyper_parameters.epochs}')
        train_loss = train_model_epoch(model, train_loader, criterion, optimizer, feature_preparer, epoch)
        test_loss = evaluate_model(model, test_loader, criterion, feature_preparer)

        # Track best test loss
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            
        # Update results
        training_results['final_train_loss'] = train_loss
        training_results['final_test_loss'] = test_loss
        training_results['best_test_loss'] = best_test_loss
        training_results['epochs_completed'] = epoch

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
        }, model_path)
        print(f'\nModel checkpointed to {model_path}')

    print(f'\nTraining completed! Best test loss: {best_test_loss:.4f}')
    return training_results


def main():
    """
    Main function for command line interface.
    Simplified interface focused on core training parameters.
    For advanced features like wandb integration, use the programmatic interface.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train HackerNews score prediction model')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training and evaluation (default: 128)')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs (default: 3)')
    parser.add_argument('--continue', type=bool, default=False,
                        help='Whether to keep training from a saved model (default: False)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate for optimizer (default: 0.001)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (default: 0.2)')
    parser.add_argument('--hidden-dim-1', type=int, default=256,
                        help='First hidden layer dimension (default: 256)')
    parser.add_argument('--hidden-dim-2', type=int, default=512,
                        help='Second hidden layer dimension (default: 512)')
    args = parser.parse_args()

    # Create ModelRunSettings using the simple constructor
    settings = ModelHyperparameters(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        hidden_dimensions=[args.hidden_dim_1, args.hidden_dim_2],
    )

    continue_training = getattr(args, "continue", False)

    # Run training with the simplified interface
    results = train_model(settings, continue_training=continue_training)
    
    print(f"\nTraining completed successfully!")
    print(f"Final results: {results}")

if __name__ == '__main__':
    main()
