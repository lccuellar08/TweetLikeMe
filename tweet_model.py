from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
import pandas as pd
 


# load

df = pd.read_csv("realdonaldtrump_formatted.csv")
df = df.dropna()
text = df.text_f.str.cat(sep=" ")


chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 280
sentences = df.text_f.tolist()

#in_filename = 'republic_sequences.txt'
doc = text #doc = load_doc(in_filename)
lines = sentences #lines = doc.split('\n')
print(max([len(s.split(" ")) for s in lines]))
 
# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences_t = tokenizer.texts_to_sequences(lines)

sequences = list()
for s in sequences_t:
	arr = [0] * 31
	for i,r in enumerate(s):
		arr[i] = r
	sequences.append(arr)

# vocabulary size
vocab_size = len(tokenizer.word_index) + 1

# separate into input and output

sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]
 
# define model
model = Sequential()
model.add(Embedding(vocab_size, 31, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X, y, batch_size=32, epochs=20)
 
# save the model to file
model.save('model.h5')
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))