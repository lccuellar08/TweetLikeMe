from numpy import array
import numpy as np
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Activation, Dropout, SimpleRNN
from helper import word_to_index, index_to_word
from keras.layers import Embedding
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras import regularizers
import pandas as pd
import sys
from gensim.models import Word2Vec

def get_data(df, screen_name, sequence_length):
	embedding_columns = [col for col in list(df.columns) if col.startswith("embedding")]

	rawX = df[embedding_columns]
	rawY = df.next_word_embedding

	X = rawX.values
	y = rawY.values

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=2020)

	return X_train, X_test, y_train, y_test

def build_model(screen_name, embeddings_model):

	pretrained_weights = embeddings_model.wv.syn0
	vocab_size, embeddings_size = pretrained_weights.shape


	model = Sequential()
	model.add(Embedding(input_dim=vocab_size, output_dim=embeddings_size, 
	                    weights=[pretrained_weights]))
	model.add(Dropout(0.4))
	model.add(SimpleRNN(units = embeddings_size, recurrent_dropout = 0.8, recurrent_regularizer = regularizers.l2(0.001)))
	model.add(Dropout(0.9))
	# model.add(LSTM(units = 256))
	# model.add(Dropout(0.9))
	# model.add(LSTM(units = 32))
	# model.add(Dropout(0.5))
	# model.add(Dense(units = 32, activation = 'relu'))
	# model.add(Dropout(0.5))
	model.add(Dense(units=vocab_size))
	model.add(Activation('softmax'))
	model.compile(optimizer = Adam(learning_rate = 0.005), loss = 'sparse_categorical_crossentropy')

	return model

def train_model(screen_name, model, X_train, X_test, y_train, y_test):
	model.fit(X_train, y_train, batch_size=128, epochs=50, validation_data = (X_test, y_test))
	model.save(screen_name+"_trained_model.h5")


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


if __name__ == '__main__':
	screen_name = sys.argv[1]
	sequence_length = 3
	embeddings_model = Word2Vec.load(screen_name + "_embeddings.bin")
	df = pd.read_csv(screen_name+"_model_input.csv")

	X_train, X_test, y_train, y_test = get_data(df, screen_name, sequence_length)
	model = build_model(screen_name, embeddings_model)
	train_model(screen_name, model, X_train, X_test, y_train, y_test)

	print(generate_next(X_test[0].tolist(), model, embeddings_model))