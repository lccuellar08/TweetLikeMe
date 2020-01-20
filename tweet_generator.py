from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
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

if __name__ == "__main__":
	pass