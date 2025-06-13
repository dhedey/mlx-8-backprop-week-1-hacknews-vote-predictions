import argparse
import os
import torch
from dateutil import parser as dateparser
from datetime import datetime, UTC

from model import HackerNewsNet, TrainingHyperparameters


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict HackerNews score prediction model')
    parser.add_argument('--title', type=str, help='The title of the story', required=True)
    parser.add_argument('--author', type=str, help='The author of the story', required=True)
    parser.add_argument('--time', type=str, help='The timestamp of the story (default = now)', default=None)
    args = parser.parse_args()

    folder = os.path.dirname(__file__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model = HackerNewsNet.load(folder, device, training_parameters=TrainingHyperparameters.for_prediction())

    title = args.title
    author = args.author
    if args.time is None:
        time = datetime.now(UTC)
    else:
        time = dateparser.parse(args.time)

    print(f"Predicting \"{title}\" by \"{author}\" at {time}...")

    prediction = model.predict(
        title=args.title,
        author=args.author,
        time=time
    )

    print(f"Prediction: {prediction}")

if __name__ == '__main__':
    main()
