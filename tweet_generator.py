from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from gensim.models import Word2Vec
from helper import word_to_index, index_to_word
import numpy as np
import pandas as pd
import random
import sys
import io

def sample(preds, temperature=1.0):
	if temperature <= 0:
		return np.argmax(preds)
	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds) / temperature
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	probas = np.random.multinomial(1, preds, 1)
	return np.argmax(probas)

def generate_next(word_idxs, model, embeddings_model, num_generated=20):
	for i in range(num_generated):
		#print(np.array(word_idxs))
		prediction = model.predict(x=np.array(word_idxs))
		idx = sample(prediction[-1], temperature=0.7)
		word_idxs.append(idx)
	return ' '.join(index_to_word(embeddings_model, idx) for idx in word_idxs)

def generate_tweet(model, embeddings_model, tweet_length, sequence_length):
	beggining_word_idx = random.sample(range(0, len(embeddings_model.wv.vocab)), sequence_length)
	words = [index_to_word(embeddings_model, word_idx) for word_idx in beggining_word_idx]

	tweet_str = words
	for i in range(tweet_length):
		next_word_prediction = model.predict(x = np.array(beggining_word_idx))
		next_word_index = sample(next_word_prediction[-1], temperature = 0.7)
		next_word = index_to_word(embeddings_model, next_word_index)
		beggining_word_idx.append(next_word_index)
		tweet_str.append(next_word)
	return ' '.join(tweet_str)

def generate_multiple_tweets(n_tweets, model, embeddings_model, sequence_length):
	for i in range(0, n_tweets):
		tweet_length = random.randint(1,30)
		print(generate_tweet(model, embeddings_model,tweet_length, sequence_length))
		print("\n")

if __name__ == "__main__":
	screen_name = sys.argv[1]
	sequence_length = 3
	embeddings_model = Word2Vec.load(screen_name + "_embeddings.bin")
	model = load_model(screen_name+"_trained_model.h5")
	generate_multiple_tweets(n_tweets = 10, model = model, embeddings_model = embeddings_model, sequence_length = sequence_length)
	# for i in range(0,10):
	# 	tweet_length = random.randint(1,30)
	# 	print(generate_tweet(model, embeddings_model,tweet_length, sequence_length))
	# 	print("\n")
