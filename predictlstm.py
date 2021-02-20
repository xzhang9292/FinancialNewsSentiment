import os
from collections import Counter
from keras.preprocessing import sequence
from keras import optimizers
import lstm as ls
import numpy as np
import tensorflow as tf
import json
from keras.models import model_from_json
from IPython.display import SVG
from keras.utils import plot_model
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot

import csv
import re
def load_model():
    json_file = open('lstm_model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
# load weights into new model
    loaded_model.load_weights("lstm_model/model.h5")
    loaded_model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    return loaded_model

def load_tsv(fname,i,j):
    day = []
    all_x = []
    all_y = []
    with open(fname,'rb') as tsvin:
        tsvin = csv.reader(tsvin,delimiter='\t')
        for row in tsvin:
            day.append(str(row[0]))
            all_x.append(str(row[i]))
            all_y.append(int(row[j]))
    return day,all_x,all_y
def precross(astring,vocabsapce = 1000, max_words = 20):
    outset = []
    for aset in astring:
        outset.append(one_hot(aset,vocabsapce))
    out = sequence.pad_sequences(outset, maxlen=max_words)
    return np.array(out)
def predict(headline,model):
	inputh = precross(headline)
	out = model.predict_classes(inputh)
	return out

if __name__ == '__main__':
	model = load_model()
	boeingday,boeingx,boeingy = load_tsv("Data/label_boeing.tsv",1,3)
	predicted = predict(boeingx,model)
	with open('predicted_boeing.tsv', 'wt') as out_file:
		tsv_writer = csv.writer(out_file, delimiter='\t')
		tsv_writer.writerow(['day','headline','true label','predicted label'])
		for i in np.arange(len(boeingx)):
			tsv_writer.writerow([boeingday[i],boeingx[i],boeingy[i],predicted[i][0]])
	msday,msx,msy = load_tsv("Data/label_microsoft.tsv",1,3)
	predicted2 = predict(msx,model)
	with open('predicted_microsoft.tsv', 'wt') as out_file2:
		tsv_writer2 = csv.writer(out_file2, delimiter='\t')
		tsv_writer2.writerow(['day','headline','true label','predicted label'])
		for i in np.arange(len(msx)):
			tsv_writer2.writerow([msday[i],msx[i],msy[i],predicted2[i][0]])









