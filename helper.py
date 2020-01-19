from gensim.models import Word2Vec

def word_to_index(embeddings_model, word):
  return embeddings_model.wv.vocab[word].index
def index_to_word(embeddings_model, idx):
  return embeddings_model.wv.index2word[idx]