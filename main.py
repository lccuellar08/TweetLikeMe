import sys
import pandas as pd
from gensim.models import Word2Vec
from keras.models import load_model
from tweet_downloader import *
from tweet_formatter import format_tweets
from tweet_embedder import prepare_data, create_embeddings
from tweet_model import get_data, build_model, train_model
from tweet_generator import generate_multiple_tweets

def main(screen_name):
	# Get all tweets for this user
	try:
		raw_tweetsDF = pd.read_csv(screen_name + "_tweets.csv")
	except:
		raw_tweetsDF = get_all_tweets(screen_name)

	# Format them
	try:
		formatted_tweetsDF = pd.read_csv(screen_name + "_formatted.csv")
	except:
		formatted_tweetsDF = format_tweets(raw_tweetsDF, screen_name)

	# Create embeddings
	sequence_length = 3
	try:
		embeddings_model = Word2Vec.load(screen_name + "_embeddings.bin")
	except:
		embeddings_model = create_embeddings(formatted_tweetsDF, screen_name, sequence_length)

	try:
		embedded_tweetsDF = pd.read_csv(screen_name + "_model_input.csv")
	except:
		embedded_tweetsDF = prepare_data(formatted_tweetsDF, screen_name, sequence_length, embeddings_model)


	# Create and Train model
	try:
		model = load_model(screen_name+"_trained_model.h5")
	except:
		X_train, X_test, y_train, y_test = get_data(embedded_tweetsDF, screen_name, sequence_length)
		model = build_model(screen_name, embeddings_model)
		train_model(screen_name, model, X_train, X_test, y_train, y_test)

	# Generate Tweets
	generate_multiple_tweets(n_tweets = 10, model = model, embeddings_model = embeddings_model, sequence_length = sequence_length)

if __name__ == "__main__":
	screen_name = sys.argv[1]
	main(screen_name)