from keras.preprocessing import sequence
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Activation

#model archtecture
def buildmodel(vocabulary_size, embedding_size, max_words):
	model=Sequential()
	model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
	model.add(LSTM(200))
	model.add(Dense(50, activation='tanh'))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	print(model.summary())
	return model




