import numpy as np
import pandas as pd
import re
import sys

#data['text'] = data['text'].apply(lambda x: x.lower())
#data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

def main(screenname):
	df = pd.read_csv(screenname+"_tweets.csv")
	for i,row in df.iterrows():
		tokens = row['text'].split(" ")
		tokens = [s for s in tokens if "http" not in s and "@" not in s]
		df.at[i,'text_f'] = " ".join(str(s) for s in tokens)
	df['text_f'] = df['text_f'].apply(lambda x: x.lower())
	df['text_f'] = df['text_f'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
	df = df[~df['text_f'].str.contains("rt")]
	df = df[df['text_f'] != '']
	print(df['text_f'])
	df.to_csv(screenname+"_formatted.csv", index=False)

if __name__ == '__main__':
	screen_name = sys.argv[1]
	main(screen_name)
