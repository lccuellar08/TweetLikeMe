import numpy as np
import pandas as pd
import re
import sys

def format_tweets(df, screen_name):
	# Remove all text that contains "http" or ""
	for i,row in df.iterrows():
		tokens = row['text'].split(" ")
		tokens = [s for s in tokens if "http" not in s and "@" not in s]
		df.at[i,'text_f'] = " ".join(str(s) for s in tokens)

	# Format all text as lowercase
	df['text_f'] = df['text_f'].apply(lambda x: x.lower())

	# Remove all non alphabet characters
	df['text_f'] = df['text_f'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

	# Remove all retweets
	df = df[~df['text_f'].str.contains("rt")]

	# Remove all empty tweets
	df = df[df['text_f'] != '']

	df.to_csv(screen_name+"_formatted.csv", index=False)

if __name__ == '__main__':
	screen_name = sys.argv[1]
	format_tweets(screen_name)
