import argparse
import os
import torch
from dateutil import parser as dateparser
from datetime import datetime, UTC
from torch.utils.data import DataLoader
import datasets
import torch.nn as nn

from model import HackerNewsNet, TrainingHyperparameters, evaluate_model, FeatureDisabling


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate HackerNews prediction')
    parser.add_argument('--no-title', type=bool, default=False)
    parser.add_argument('--no-author', type=bool, default=False)
    parser.add_argument('--no-domain', type=bool, default=False)
    parser.add_argument('--no-year', type=bool, default=False)
    parser.add_argument('--no-day', type=bool, default=False)
    parser.add_argument('--no-time', type=bool, default=False)
    args = parser.parse_args()

    folder = os.path.dirname(__file__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    feature_disabling = FeatureDisabling(
        disable_title=args.no_title,
        disable_domain=args.no_domain,
        disable_author=args.no_author,
        disable_year=args.no_year,
        disable_day=args.no_day,
        disable_time=args.no_time,
    )
    model = HackerNewsNet.load(
        folder,
        device,
        training_parameters=TrainingHyperparameters.for_prediction(),
        feature_disabling=feature_disabling,
    )

    # Load the filtered dataset
    dataset = datasets.load_from_disk(folder + "/filtered_dataset")
    print(f"Dataset loaded, size: {len(dataset)}")

    (train_dataset, test_dataset) = torch.utils.data.random_split(
        dataset,
        [0.9, 0.1],
        torch.Generator().manual_seed(42)
    )
    print(f"Dataset loaded and split into train: {len(train_dataset)} and test: {len(test_dataset)}")

    criterion = nn.MSELoss()  # MSE loss for predicting log scores
    test_loader = DataLoader(test_dataset, batch_size=1024)

    print(feature_disabling)
    print("Running evaluation...")
    evaluate_model(model, test_loader, criterion, should_print=True)

if __name__ == '__main__':
    main()
