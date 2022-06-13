import nltk

nltk.download('punkt')
import numpy as np
from nltk.stem.lancaster import LancasterStemmer

from NeuralNetwork import NeuralNetwork
from Repository import Repository

stemmer = LancasterStemmer()
import random


class Main:
    def __init__(self):
        self.repository = Repository()
        self.words, self.labels, self.trainingList, self.outputList = self.repository.preprocessingToIntent()
        self.neuralNetwork = NeuralNetwork(self.trainingList, self.outputList)
        self.neuralNetwork.getModel(self.trainingList, self.outputList)

    def bag_of_words(self, s, words):
        bag = [0 for _ in range(len(words))]
        s_words = nltk.word_tokenize(s)
        s_words = [stemmer.stem(word.lower()) for word in s_words]

        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    bag[i] = 1
        return np.array(bag)

    def chat(self):
        print('Merhaba ben konuşabiliyorum! (Çıkmak için q tuşuna basınız) !')
        while True:
            inp = input('You: ')
            if inp.lower() == 'q':
                break
            results = self.neuralNetwork.model.predict([self.bag_of_words(inp, self.words)])[0]
            ## To display the best answer get largest probablity
            results_index = np.argmax(results)
            tag = self.labels[results_index]
            print(results[results_index])
            if results[results_index] > 0.9:
                for tg in self.repository.data['intents']:
                    if tg['tag'] == tag:
                        responses = tg['responses']

                print(random.choice(responses))
            else:
                for tg in self.repository.data['intents']:
                    if tg['tag'] == tag:
                        responses = tg['responses']
                print("Ne demek istediğinizi tam anlamadım. Şöyle bir cevap vericem umarım absürt değildir :) " + random.choice(responses))


main = Main()
main.chat()
