# encoding: utf-8
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

def load_tsv(fname,i,j):
    all_x = []
    all_y = []
    with open(fname,'rb') as tsvin:
        tsvin = csv.reader(tsvin,delimiter='\t')
        for row in tsvin:
            all_x.append(str(row[i]))
            all_y.append(int(row[j]))
    total = len(all_x)
    sample = int(total*0.2)
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    random_index = np.random.randint(len(all_x),size = sample)
    random_index = np.array(random_index)
    test_x = all_x[random_index]
    test_y = all_y[random_index]
    train_x = np.delete(all_x,random_index)
    train_y = np.delete(all_y,random_index)

    return train_x, train_y, test_x, test_y



def precross(astring,vocabsapce, max_words):
    outset = []
    for aset in astring:
        outset.append(one_hot(aset,vocabsapce))
    out = sequence.pad_sequences(outset, maxlen=max_words)

    return np.array(out)
#load for comment

def save_model(model):
    print("Saving model")
    model_json = model.to_json()
    with open("lstm_model/model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("lstm_model/model.h5")
    json_file.close()

def buildingModel():
    max_words = 20
    vocabspace = 1000
    embedding_size = 3
    batch_size = 100
    num_epochs = 30
    np.random.seed(1)
    tf.set_random_seed(1)

    #load boeing data
    train_x1, train_y1, test_x1, test_y1 = load_tsv('Data/label_boeing.tsv',1,3)
    train_x1 = precross(train_x1,vocabspace, max_words)
    test_x1 = precross(test_x1,vocabspace,max_words)
    #load microsoft data
    train_x2, train_y2, test_x2, test_y2 = load_tsv('Data/label_microsoft.tsv',1,3)
    train_x2 = precross(train_x2,vocabspace, max_words)
    test_x2 = precross(test_x2,vocabspace,max_words)

    #combine two training data
    train_x = np.concatenate((train_x2,train_x1))
    train_y = np.concatenate((train_y2,train_y1))
    random_index = np.random.randint(len(train_x),size = len(train_x))
    random_index = np.array(random_index)
    test_x = train_x[random_index]
    test_y = train_y[random_index]


    #build train model
    model = ls.buildmodel(vocabspace,embedding_size,max_words)
    opt = optimizers.RMSprop(lr=0.001, rho=0.8, epsilon=None, decay=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.fit(train_x, train_y, batch_size=batch_size, shuffle=False, epochs=num_epochs, verbose=0)
    scores,accuracy = model.evaluate(train_x1, train_y1, verbose=0)
    #print("Accuracy for training Boeing: %2f",accuracy)
    scores,accuracy = model.evaluate(train_x2, train_y2, verbose=0)
    #print("Accuracy for training Microsoft: %2f",accuracy)
    scores2,accuracy2 = model.evaluate(test_x1, test_y1, verbose=0)
    print("Accuracy for Boeing %2f",accuracy2)
    scores3,accuracy3 = model.evaluate(test_x2,test_y2, verbose=0)
    print("Accuracy for Microsoft %2f",accuracy3)
    save_model(model)
    return model

def load_model():
    json_file = open('lstm_model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
# load weights into new model
    loaded_model.load_weights("lstm_model/model.h5")
    loaded_model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    return loaded_model

if __name__ == '__main__':
    model= buildingModel()
    #model= load_model()
    #x = load_prediction_data(vocab_to_int)
    #x = np.array(x)
    #out1 = model.predict_classes(x)







