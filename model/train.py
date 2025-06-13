import argparse
import os
from pprint import pprint

import datasets
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from model import TrainingHyperparameters, ModelHyperparameters, HackerNewsNet, FeaturePreparer

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

def train_model(
        model_parameters: ModelHyperparameters,
        training_parameters: TrainingHyperparameters,
        continue_training = False
    ):
    
    # Load configuration
    load_dotenv()
    folder = os.path.dirname(__file__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print("Hyperparameters:")
    pprint(model_parameters.to_dict())

    print("Training parameters:")
    pprint(training_parameters.to_dict())

    feature_preparer = FeaturePreparer.load(folder, device)

    print("Feature parameters:")
    pprint(feature_preparer.feature_parameters().to_dict())

    print("Loading dataset...")
    datasets.config.IN_MEMORY_MAX_SIZE = 8 * 1024 * 1024 # 8GB

    # Load the filtered dataset
    dataset = datasets.load_from_disk(folder + "/filtered_dataset")
    print(f"Dataset loaded, size: {len(dataset)}")

    (train_dataset, test_dataset) = torch.utils.data.random_split(
        dataset,
        [0.9, 0.1],
        torch.Generator().manual_seed(42)
    )
    print(f"Dataset loaded and split into train: {len(train_dataset)} and test: {len(test_dataset)}")

    # Get train and test loaders
    train_loader = DataLoader(train_dataset, batch_size=training_parameters.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=training_parameters.batch_size)

    model_path = folder + '/hackernews_model.pth'

    # Initialize model
    model = HackerNewsNet(feature_preparer, model_parameters, training_parameters).to(device)
    optimizer = optim.Adam(model.parameters(), lr=training_parameters.learning_rate)

    start_epoch = 1
    if continue_training:
        print(f"Loading model and optimizer state from {model_path}...")
        try:
            loaded_data = torch.load(model_path)
            loaded_model_parameters = loaded_data["model_parameters"]
            if loaded_model_parameters != model_parameters:
                raise RuntimeError("Loaded model parameters do not match provided model parameters. Cannot continue training.")
            optimizer.load_state_dict(loaded_data['optimizer'])
            model.load_state_dict(loaded_data["model"])
            start_epoch = loaded_data['epoch']
        except (FileNotFoundError, KeyError) as e:
            print(f"Could not load model: {e}. Starting from scratch.")

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

    for epoch in range(start_epoch, training_parameters.epochs + 1):
        print(f'\nEpoch {epoch}/{training_parameters.epochs}')
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
            'model_parameters': model_parameters.to_dict(),
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
    parser.add_argument('--learning-rate', type=float, default=0.002,
                        help='Learning rate for optimizer (default: 0.002)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (default: 0.3)')
    parser.add_argument('--hidden-dim-1', type=int, default=256,
                        help='First hidden layer dimension (default: 256)')
    parser.add_argument('--hidden-dim-2', type=int, default=1024,
                        help='Second hidden layer dimension (default: 1024)')
    args = parser.parse_args()

    continue_training = getattr(args, "continue", False)

    # Run training with the simplified interface
    results = train_model(
        model_parameters=ModelHyperparameters(
            hidden_dimensions=[args.hidden_dim_1, args.hidden_dim_2],
        ),
        training_parameters=TrainingHyperparameters(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            dropout=args.dropout,
            freeze_embeddings=False,
        ),
        continue_training=continue_training,
    )
    
    print(f"\nTraining completed successfully!")
    print(f"Final results: {results}")

if __name__ == '__main__':
    main()
