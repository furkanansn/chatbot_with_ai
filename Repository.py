import nltk
import numpy as np
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()
import json
import pickle


class Repository:
    def __init__(self):
        with open("intents.json") as file:
            self.data = json.load(file)

    def preprocessingToIntent(self):
        words = []
        labels = []
        docs_x = []
        docs_y = []
        for intent in self.data['intents']:
            for pattern in intent['patterns']:
                ## seprating the word into a list
                wrds = nltk.word_tokenize(pattern)
                ## adding all to the list
                words.extend(wrds)
                ## append the tokenize words
                docs_x.append(wrds)
                ## append the tag of the intent
                docs_y.append(intent['tag'])
            if intent['tag'] not in labels:
                labels.append(intent['tag'])
        ## lowering the words to avoid confusing
        ## removing the ?
        words = [stemmer.stem(w.lower()) for w in words if w not in '?']
        ## Removing all the duplicates
        words = sorted(list(set(words)))

        labels = sorted(labels)

        ## creating a bagged of words in binary to train the model
        ## So we can do one hot-encoding with the words
        training = []
        output = []

        ## list of tags into one hot-encoding
        out_empty = [0 for _ in range(len(labels))]

        for x, doc in enumerate(docs_x):
            bag = []
            wrds = [stemmer.stem(w) for w in doc]

            for w in words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)
            output_row = out_empty[:]
            output_row[labels.index(docs_y[x])] = 1
            training.append(bag)
            output.append(output_row)
        ## switching the list into an array for input into a model
        training = np.array(training)
        output = np.array(output)
        ## saving the preprocessing
        with open('data.pickle', 'wb') as f:
            pickle.dump((words, labels, training, output), f)
        return words, labels, training, output
