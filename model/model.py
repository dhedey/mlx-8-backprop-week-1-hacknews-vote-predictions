import datetime
import re
from collections import OrderedDict
from dataclasses import dataclass, field

import math
import pandas as pd
import tldextract
import torch
import torch.nn as nn


def extract_domain(url):
    if not isinstance(url, str):
        return None

    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    extracted = tldextract.extract(url)

    # Return domain.suffix (e.g., 'google.com', 'blogspot.com')
    if extracted.domain and extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}".lower()

    return None

@dataclass
class TrainingHyperparameters:
    batch_size: int
    epochs: int
    learning_rate: float
    freeze_embeddings: bool
    dropout: float

    @classmethod
    def for_prediction(cls):
        return cls(
            batch_size=1,
            epochs=0,
            learning_rate=0,
            freeze_embeddings=True,
            dropout=0,
        )

    def to_dict(self):
        return vars(self)


@dataclass
class ModelHyperparameters:
    domain_embedding_size: int = 16
    author_embedding_size: int = 16
    hidden_dimensions: list[int] = field(default_factory=lambda: [256, 128, 64])
    include_batch_norms: bool = False

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

@dataclass
class FeatureDisabling:
    disable_title: bool = False
    disable_author: bool = False
    disable_domain: bool = False
    disable_year: bool = False
    disable_day: bool = False
    disable_time: bool = False

    def to_dict(self):
        return vars(self)

def create_id_map(df, min_count, column):
    return {
        x: i for i, x in enumerate(df[df["count"] >= min_count][column])
    }


class FeaturePreparer:
    @classmethod
    def load(cls, folder, device):
        word_vectors = torch.load(folder + '/word_vectors.pt')
        return cls(
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

    def __init__(self, domain_map, author_map, vocabulary, vocabulary_embeddings, device, title_token_length=20):
        self.domain_map = domain_map
        self.author_map = author_map
        self.title_vocab_map = {word: i for i, word in enumerate(vocabulary)}
        self.initial_vocabulary_embeddings = torch.cat([
            vocabulary_embeddings,
            torch.zeros((1, vocabulary_embeddings.shape[1]), dtype=torch.float32)  # Placeholder padding word
        ]).to(device)
        assert vocabulary_embeddings.shape[0] == len(vocabulary)
        self.title_token_length = title_token_length
        self.device = device

    def _tokenize_title(self, title):
        filtered_title_words = re.sub(r'[^a-z0-9 ]', '', title.lower()).split()
        token_placeholder = len(self.title_vocab_map)  # Placeholder
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
            return len(self.author_map)  # Unknown

    def _map_domain_id(self, domain):
        if domain in self.domain_map:
            return self.domain_map[domain]
        else:
            return len(self.domain_map)  # Unknown

    def feature_parameters(self) -> FeatureParameters:
        return FeatureParameters(
            title_token_ids=len(self.title_vocab_map) + 1,  # Include placeholder
            title_embedding_size=self.initial_vocabulary_embeddings.shape[1],
            title_token_length=self.title_token_length,  # Anything more than this will be truncated, or padded
            author_token_ids=len(self.author_map) + 1,  # Include unknown
            domain_token_ids=len(self.domain_map) + 1,  # Include unknown
            time_features=5,  # year, day of week cos/sin, hour of day cos/sin
        )

    def prepare_batch_features(self, batch):
        tokenized_titles = [self._tokenize_title(title) for title in batch['title']]

        author_ids = [self._map_author_id(author) for author in batch['author']]
        domain_ids = [self._map_domain_id(domain) for domain in batch['domain']]

        timestamps = [datetime.datetime.fromtimestamp(time, datetime.UTC) for time in batch['time']]
        year = [date.year - 2000 for date in timestamps]
        day_of_week_cos = [math.cos(2 * math.pi * date.weekday() / 7) for date in timestamps]
        day_of_week_sin = [math.sin(2 * math.pi * date.weekday() / 7) for date in timestamps]
        hour_of_day_cos = [math.cos(2 * math.pi * date.hour / 24) for date in timestamps]
        hour_of_day_sin = [math.sin(2 * math.pi * date.hour / 24) for date in timestamps]

        return {
            'tokenized_titles': torch.tensor(tokenized_titles, dtype=torch.int).to(self.device),
            'author_id': torch.tensor(author_ids, dtype=torch.int).to(self.device),
            'domain_id': torch.tensor(domain_ids, dtype=torch.int).to(self.device),
            'time': torch.stack(
                [
                    torch.tensor(year, dtype=torch.int).to(self.device),
                    torch.tensor(day_of_week_cos, dtype=torch.int).to(self.device),
                    torch.tensor(day_of_week_sin, dtype=torch.int).to(self.device),
                    torch.tensor(hour_of_day_cos, dtype=torch.int).to(self.device),
                    torch.tensor(hour_of_day_sin, dtype=torch.int).to(self.device),
                ],
                dim=1,
            ).to(self.device),
        }

    def prepare_batch(self, batch):
        return {
            'ids': torch.tensor([id for id in batch['id']], dtype=torch.long).to(self.device),
            'log_scores': torch.log(torch.tensor([score for score in batch['score']], dtype=torch.float32) + 1).to(
                self.device),
            'features': self.prepare_batch_features(batch),
        }

class HackerNewsNet(nn.Module):
    @classmethod
    def load(cls, folder, device, training_parameters, feature_disabling: FeatureDisabling = None):
        torch.serialization.add_safe_globals([ModelHyperparameters])
        model_location = folder + '/hackernews_model.pth'

        print(f"Location {model_location}")
        loaded_data = torch.load(model_location, map_location=device)

        model = cls(
            feature_preparer = FeaturePreparer.load(folder, device),
            model_params = loaded_data["model_parameters"],
            training_params = training_parameters,
            feature_disabling = feature_disabling,
        ).to(device)

        model.load_state_dict(loaded_data["model"])
        return model

    def __init__(self, feature_preparer: FeaturePreparer, model_params: ModelHyperparameters,
                 training_params: TrainingHyperparameters, feature_disabling: FeatureDisabling = None):
        super(HackerNewsNet, self).__init__()

        self.disabling = feature_disabling or FeatureDisabling()
        self.feature_preparer = feature_preparer
        feature_parameters = feature_preparer.feature_parameters()

        self.title_embedding = nn.Embedding(
            feature_parameters.title_token_ids,
            feature_parameters.title_embedding_size,
            _weight=feature_preparer.initial_vocabulary_embeddings,
            _freeze=training_params.freeze_embeddings,
        )
        self.author_embedding = nn.Embedding(feature_parameters.author_token_ids, model_params.author_embedding_size)
        self.domain_embedding = nn.Embedding(feature_parameters.domain_token_ids, model_params.domain_embedding_size)

        input_feature_length = feature_parameters.title_embedding_size + model_params.author_embedding_size + model_params.domain_embedding_size + feature_parameters.time_features

        self.device = feature_preparer.device

        # Layers
        input_layer_sizes = [input_feature_length] + model_params.hidden_dimensions[:-1]
        output_layer_sizes = model_params.hidden_dimensions

        self.inner_layers = nn.Sequential(OrderedDict([
            (f'layer_{i + 1}', nn.Sequential(OrderedDict([
                ('linear', nn.Linear(input_size, output_size)),
                ('relu', nn.ReLU()),
                ('batch_norm', nn.BatchNorm1d(output_size) if model_params.include_batch_norms else nn.Identity()),
                ('dropout', nn.Dropout(p=training_params.dropout)),
            ])))
            for i, (input_size, output_size) in enumerate(zip(input_layer_sizes, output_layer_sizes))
        ]))
        self.prediction_layer = nn.Linear(model_params.hidden_dimensions[-1], 1)

    def predict(self, title, url, author, time):
        self.eval()
        domain = extract_domain(url) or "UNKNOWN"

        prepared_batch_features = self.feature_preparer.prepare_batch_features({
            "title": [title],
            "domain": [domain],
            "author": [author],
            "time": [time.timestamp()]
        })
        with torch.no_grad():
            output = self.forward(prepared_batch_features)

        score = torch.exp(output[0]) - 1
        return {
            "log_score": output[0].detach().item(),
            "score": score.detach().item(),
        }

    def forward(self, features):
        embedded_title_tokens = self.title_embedding(
            features['tokenized_titles'])  # (batch, title_length, embedding_dim)

        # We now create features which should be of dimension (batch, feature_length)
        title_features = torch.mean(embedded_title_tokens, dim=1)
        if self.disabling.disable_title:
            title_features = torch.zeros_like(title_features)

        time_features = features['time']
        if self.disabling.disable_year:
            time_features[:, 0] = 0
        if self.disabling.disable_day:
            time_features[:, 1] = 0
            time_features[:, 2] = 0
        if self.disabling.disable_time:
            time_features[:, 3] = 0
            time_features[:, 4] = 0

        domain_features = self.domain_embedding(features['domain_id'])
        if self.disabling.disable_domain:
            domain_features = torch.zeros_like(domain_features)

        author_features = self.author_embedding(features['author_id'])
        if self.disabling.disable_author:
            author_features = torch.zeros_like(author_features)

        x = torch.cat(  # Concatenate along the feature dimension
            [
                title_features,
                domain_features,
                author_features,
                time_features,
            ],
            dim=1,
        )

        x = self.inner_layers(x)
        x = self.prediction_layer(x)    # Final linear layer to get the output, without dropout / batch norm

        return torch.squeeze(x, dim=1)  # The final dimension is size 1 - flatten it / remove it


def evaluate_model(model: HackerNewsNet, data_loader, criterion, should_print=True):
    """
    Evaluate the model on test data.
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0
    predicted_log_scores = []
    actual_log_scores = []

    if should_print:
        print("Evaluating on test data...")

    with torch.no_grad():
        for raw_batch in data_loader:
            # Process the batch
            batch = model.feature_preparer.prepare_batch(raw_batch)
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

    if should_print:
        print(f'Test Loss (MSE on log scores): {avg_loss:.4f}')
        print(f'Mean predicted log score: {predicted_log_scores.mean():.4f} (std: {predicted_log_scores.std():.4f})')
        print(f'Mean actual    log score: {actual_log_scores.mean():.4f} (std: {actual_log_scores.std():.4f})')
        print(f'Mean predicted raw score: {predicted_raw_scores.mean():.2f} (std: {predicted_raw_scores.std():.4f})')
        print(f'Mean actual    raw score: {actual_raw_scores.mean():.4f} (std: {actual_raw_scores.std():.4f})')

    return {
        "test_loss": avg_loss,
        "predicted_log_score_mean": predicted_log_scores.mean().item(),
        "predicted_log_score_std": predicted_log_scores.std().item(),
        "actual_log_score_mean": actual_log_scores.mean().item(),
        "actual_log_score_std": actual_log_scores.std().item(),
        "predicted_raw_score_mean": predicted_raw_scores.mean().item(),
        "predicted_raw_score_std": predicted_raw_scores.std().item(),
        "actual_raw_score_mean": actual_raw_scores.mean().item(),
        "actual_raw_score_std": actual_raw_scores.std().item(),
    }