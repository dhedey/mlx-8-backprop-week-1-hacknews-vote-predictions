import gensim
from gensim.models import Word2Vec
import logging
import os
import numpy as np
import torch

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def train_word2vec():
    # Read the text8 file
    sentences = gensim.models.word2vec.LineSentence('text8.txt')
    
    # Train the model with optimized parameters
    model = Word2Vec(sentences,
                    vector_size=300,    # dimension of word vectors
                    window=5,           # context window size
                    min_count=5,        # ignore words that appear less than this
                    workers=4,          # number of CPU cores to use
                    sg=0,              # use cbow (not skip-gram) model
                    epochs=3,          # reduced from 5 to 3 epochs
                    batch_words=10000)  # increased batch size for faster training
    
    # Extract vocabulary and embeddings
    vocabulary = list(model.wv.index_to_key)  # Get all words in vocabulary
    embeddings = model.wv.vectors  # Get the word vectors
    
    # Convert to PyTorch tensors
    embeddings_tensor = torch.FloatTensor(embeddings)
    
    # Save vocabulary and embeddings in PyTorch format
    torch.save({
        'vocabulary': vocabulary,
        'embeddings': embeddings_tensor
    }, 'word_vectors.pt')
    
    logging.info(f"Saved vocabulary of {len(vocabulary)} words and their embeddings")
    logging.info(f"Vector shape: {embeddings_tensor.shape}")
    
    # Print some example words and their most similar words
    example_words = ['computer', 'programming', 'data', 'ai']
    for word in example_words:
        try:
            similar = model.wv.most_similar(word, topn=3)
            logging.info(f"\n{word}:")
            for similar_word, similarity in similar:
                logging.info(f"  {similar_word}: {similarity:.4f}")
        except KeyError:
            logging.info(f"\n{word} not found in vocabulary")
    
    return vocabulary, embeddings_tensor

def load_word_vectors(device=None):
    """
    Load the saved word vectors and vocabulary
    Args:
        device: torch.device to load tensors to (e.g., 'cuda' for GPU)
    """
    try:
        # Load the saved data
        saved_data = torch.load('word_vectors.pt')
        vocabulary = saved_data['vocabulary']
        embeddings = saved_data['embeddings']
        
        # Move to specified device if provided
        if device is not None:
            embeddings = embeddings.to(device)
        
        # Create dictionary mapping words to their vectors
        word_vectors = {word: vector for word, vector in zip(vocabulary, embeddings)}
        
        return word_vectors
    
    except Exception as e:
        logging.error(f"Error loading word vectors: {e}")
        raise

if __name__ == "__main__":
    vocabulary, embeddings = train_word2vec()
    print("Word vectors saved as 'word_vectors.pt'")
    
    # Print file size
    pt_size = os.path.getsize('word_vectors.pt') / (1024 * 1024)  # Size in MB
    print(f"\nFile size: {pt_size:.2f} MB") 