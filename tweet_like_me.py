import sys
from tweet_downloader import *
from tweet_formatter import format_tweets
from tweet_embedder import prepare_data, create_embeddings
from tweet_model import get_data, build_model, train_model, generate_next

def main(screen_name):
	# Get all tweets for this user
	raw_tweetsDF = get_all_tweets(screen_name)

	# Format them
	formatted_tweetsDF = format_tweets(raw_tweetsDF, screen_name)

	# Create embeddings
	sequence_length = 3
	embeddings_model = create_embeddings(formatted_tweetsDF, screen_name, sequence_length)
	embedded_tweetsDF = prepare_data(formatted_tweetsDF, screen_name, sequence_length, embeddings_model)

	# Create and Train model
	X_train, X_test, y_train, y_test = get_data(embedded_tweetsDF, screen_name, sequence_length)
	model = build_model(screen_name, embeddings_model)
	train_model(screen_name, model, X_train, X_test, y_train, y_test)

	#print(generate_next(model, embeddings_model))


if __name__ == "__main__":
	screen_name = sys.argv[1]
	main(screen_name)
	#richieofarrill_
	#print(generate_next(X_test[0].tolist(), model, embeddings_model))