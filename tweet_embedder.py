import sys
import pandas as pd
import numpy as np
from helper import word_to_index, index_to_word
from gensim.models import Word2Vec


def prepare_data(screen_name, sequence_length, embeddings_model):
	df = pd.read_csv(screen_name+"_formatted.csv")
	tweets = df.text_f.values.tolist()
	split_tweets = [tweet.split(" ") for tweet in tweets]

	split_tweets = [tweet.split(" ") for tweet in tweets if len(tweet.split(" ")) > sequence_length]

	# Now for every tweet, let's create this dataframe such as:
	#  ___________________________________
	# | TWEET SEQUENCE 	     | NEXT WORD  | 
	# |----------------------|------------|
	# | ['this','is','a'] 	 | 'tweet'    |
	# | ['is','a', 'tweet']  | 'of'		  |
	# | ['a', 'tweet' ,'of'] | 'multiple' |
	# |______________________|____________|

	tweet_records = []
	for tweet in split_tweets:
		for i in range(0, len(tweet) - sequence_length):
			tweet_sequence = tweet[i: i + sequence_length]
			# print(tweet_sequence)

			try:
				tweet_sequence_embedding = [word_to_index(embeddings_model, word) for word in tweet_sequence]
			except:
				# print(tweet_sequence)
				# print(i)
				sys.exit(0)

			next_word = tweet[i + sequence_length]
			next_word_embedding = word_to_index(embeddings_model, next_word)

			tweet_records.append({'tweet_sequence': tweet_sequence, 'tweet_sequence_embedding': tweet_sequence_embedding,
				'next_word': next_word, 'next_word_embedding': next_word_embedding})
	tweetsDF = pd.DataFrame(tweet_records)
	tweetsDF = tweetsDF[['tweet_sequence', 'tweet_sequence_embedding', 'next_word', 'next_word_embedding']]

	

	# input_words = np.zeros([len(split_tweets), sequence_length], dtype=np.int32)
	# output_word = np.zeros([len(sentences)], dtype=np.int32)


	# for i, sentence in enumerate(sentences):
	#   for t, word in enumerate(sentence[:-1]):
	#     train_x[i, t] = word2idx(word)
	#   train_y[i] = word2idx(sentence[-1])


	tweetsDF.to_csv(screen_name+"_model_input.csv", index = False)


def create_embeddings(screen_name, sequence_length):
	df = pd.read_csv(screen_name+"_formatted.csv")
	tweets = df.text_f.values.tolist()
	split_tweets = [tweet.split(" ") for tweet in tweets]
	model = Word2Vec(split_tweets, min_count = 3)
	# print(model.wv.index2word)

	model.save(screen_name+"_embeddings.bin")

	return(model)

if __name__ == '__main__':
	screen_name = sys.argv[1]
	sequence_length = 3
	model = create_embeddings(screen_name, sequence_length)
	prepare_data(screen_name, sequence_length, model)