import sys
import pandas as pd
import numpy as np
from helper import word_to_index, index_to_word
from gensim.models import Word2Vec

def create_embeddings(df, screen_name, sequence_length):
	tweets = df.text_f.values.tolist()
	split_tweets = [tweet.split(" ") for tweet in tweets if len(tweet.split(" ")) > sequence_length]
	model = Word2Vec(split_tweets, size = 100, window = sequence_length - 1, sg = 1, min_count = 1)

	model.save(screen_name+"_embeddings.bin")

	return(model)

def prepare_data(df, screen_name, sequence_length, embeddings_model):
	tweets = df.text_f.values.tolist()
	split_tweets = [tweet.split(" ") for tweet in tweets if len(tweet.split(" ")) > sequence_length]

	tweet_records = []
	embedding_columns = ["embedding_"+str(i) for i in range(0,sequence_length)]

	for tweet in split_tweets:
		for i in range(0, len(tweet) - sequence_length):
			tweet_sequence = tweet[i: i + sequence_length]

			tweet_sequence_embedding = [word_to_index(embeddings_model, word) for word in tweet_sequence]
			tweet_sequence_embedding = dict(zip(embedding_columns, tweet_sequence_embedding))

			next_word = tweet[i + sequence_length]
			next_word_embedding = word_to_index(embeddings_model, next_word)

			record = {'tweet_sequence': tweet_sequence, 'next_word': next_word, 'next_word_embedding': next_word_embedding}
			record.update(tweet_sequence_embedding)

			tweet_records.append(record)

	tweetsDF = pd.DataFrame(tweet_records)
	tweetsDF = tweetsDF[embedding_columns + ['tweet_sequence', 'next_word', 'next_word_embedding']]

	tweetsDF.to_csv(screen_name+"_model_input.csv", index = False)

	return(tweetsDF)


if __name__ == '__main__':
	screen_name = sys.argv[1]
	sequence_length = 3
	df = pd.read_csv(screen_name+"_formatted.csv")

	model = create_embeddings(df, screen_name, sequence_length)
	prepare_data(df, screen_name, sequence_length, model)