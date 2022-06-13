import tflearn
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


class NeuralNetwork:
    def __init__(self,trainingList,outputList):
        self.defineInputShapeForModel(trainingList)
        self.defineHiddenLayers()
        self.defineActivationForNeurons(outputList)
        self.model = ""

    def defineInputShapeForModel(self,trainingList):
        self.net = tflearn.input_data(shape=[None, len(trainingList[0])])

    def defineHiddenLayers(self):
        self.net = tflearn.fully_connected(self.net, 8)
        self.net = tflearn.fully_connected(self.net, 8)

    def defineActivationForNeurons(self,outputList):
        self.net = tflearn.fully_connected(self.net, len(outputList[0]), activation='softmax')
        self.net = tflearn.regression(self.net)

    def getModel(self,trainingList, outputList):
        self.model = tflearn.DNN(self.net)
        self.model.fit(trainingList, outputList, n_epoch=1000, batch_size=8, show_metric=True)
        self.model.save('model.tflearn')