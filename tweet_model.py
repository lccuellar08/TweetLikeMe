import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt

from helper import word_to_index, index_to_word
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec

from keras.models import Sequential
from keras.layers import Activation, Dropout, SimpleRNN, Dense, Embedding
from keras.optimizers import SGD
from keras import regularizers

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
	model.add(SimpleRNN(units = embeddings_size, recurrent_dropout = 0.7, recurrent_regularizer = regularizers.l2(0.001)))
	model.add(Dropout(0.8))
	model.add(Dense(units=vocab_size))
	model.add(Activation('softmax'))
	optimizer = SGD(lr = 0.005, decay = 1e-9, momentum = 0.90)
	model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy')

	return model

def train_model(screen_name, model, X_train, X_test, y_train, y_test):
	n_epochs = 500
	history = model.fit(X_train, y_train, batch_size=32, epochs=n_epochs, validation_data = (X_test, y_test))

	plt.plot(range(0, n_epochs), history.history['loss'], label = "loss")
	plt.plot(range(0, n_epochs), history.history['val_loss'], label = "val_loss")
	plt.legend()
	plt.show()

	model.save(screen_name+"_trained_model.h5")


if __name__ == '__main__':
	screen_name = sys.argv[1]
	sequence_length = 3
	embeddings_model = Word2Vec.load(screen_name + "_embeddings.bin")
	df = pd.read_csv(screen_name+"_model_input.csv")

	X_train, X_test, y_train, y_test = get_data(df, screen_name, sequence_length)
	model = build_model(screen_name, embeddings_model)
	train_model(screen_name, model, X_train, X_test, y_train, y_test)

	print(generate_next(X_test[0].tolist(), model, embeddings_model))