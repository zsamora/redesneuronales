import numpy as np
import unittest
import matplotlib.pyplot as plt
import random
import math

class AbstractNeuron:
    def __init__(self, w, b, lr=0.1, precision = 0):
        self.weight = w # Vector of weights
        self.bias = b   # Bias
        self.lr = lr # Learning rate
        self.precision = precision
        ## Clase 22/10 Learning rate = 0.5
    def train(self, point, expected):
        real = self.feed(point) # Perceptron's output according to w and b
        if (expected != real):
            diff = expected - real
            newweight = self.weight + (self.lr * point * diff) # Este se puede repetir en todos los input en vez de realizarlo solo una vez)
            self.weight = newweight
            newbias = self.bias + (self.lr * diff)
            self.bias = newbias
    def plotpoints(self):
        tpoints = input('Ingrese la cantidad de puntos para entrenar: ')
        tpoints = int(tpoints)
        npoints = input('Ingrese la cantidad de puntos a graficar: ')
        npoints = int(npoints)
        trainingpoints = np.random.rand(tpoints,2)
        for t in trainingpoints:
            if t[0] < t[1]: # Recta y = x
                self.train(t,0)
            else:
                self.train(t,1)
        ## Clasificacion de los puntos reales
        newpoints = np.random.rand(npoints,2)
        listclassification = []
        for n in newpoints:
            if n[0] < n[1]: # Recta y = x
                listclassification.append(0)
            else:
                listclassification.append(1)
        output = []
        for po in range(0,len(newpoints)):
            classification = self.feed(newpoints[po])
            output.append(classification)
            if classification:
                plt.scatter(newpoints[po][0],newpoints[po][1],c='red')
            else:
                plt.scatter(newpoints[po][0],newpoints[po][1],c='blue')
        plt.plot([0,1], [0,1],c='black')
        aciertos = 0
        for l,o in zip(listclassification,output):
            aciertos += 1 if (l == o) else 0
        precision = aciertos / len(output)
        print(precision)
        plt.show()
    def plotlearning(self):
        p = []
        npoints = input('Ingrese la cantidad de puntos a graficar: ')
        npoints = int(npoints)
        for i in range (0,500):
            p.append(self.precisionlearning(i,npoints))
        plt.plot(range(0,500),p)
        plt.show()

    def precisionlearning(self, tpoints, npoints):
        trainingpoints = np.random.rand(tpoints,2)
        for t in trainingpoints:
            if t[0] < t[1]: # Recta y = x
                self.train(t,0)
            else:
                self.train(t,1)
        ## Clasificacion de los puntos reales
        newpoints = np.random.rand(npoints,2)
        listclassification = []
        for n in newpoints:
            if n[0] < n[1]: # Recta y = x
                listclassification.append(0)
            else:
                listclassification.append(1)
        output = []
        for po in range(0,len(newpoints)):
            classification = self.feed(newpoints[po])
            output.append(classification)
        aciertos = 0
        for l,o in zip(listclassification,output):
            aciertos += 1 if (l == o) else 0
        precision = aciertos / len(output)
        return precision

class Perceptron(AbstractNeuron):
    def feed(self, x):
        result = np.dot(self.weight, x) + self.bias
        if (result <= 0):
            return 0
        else:
            return 1

class Sigmoid(AbstractNeuron):
    def feed(self, x):
        result = 1 / (1 + math.exp(-1 * np.dot(self.weight, x) - self.bias))
        return result

class NeuronLayer():
    def __init__(self, nneurons=1, isoutput=True, ninput=0):
        self.isoutput = isoutput
        self.nneurons = nneurons
        self.neuronarray = []
    def createNeurons(self, nneurons, ninput):
        res = []
        while (nneurons > 0):
            p = Perceptron(np.random.rand(ninput)*2, random.randint(-2,2), 0.1)
            res.append(p)
            nneurons = nneurons - 1
        return res

    def feed (self, input):
        output = []
        for n in self.neuronarray:
            output.append(n.feed(input))
        return output

class NeuralNetwork():
    def __init__(self, nlayers=1, neuronsarray=[1], ninput=1):
        self.nlayers = nlayers
        self.narray = neuronsarray
        self.ninput = ninput
        self.layerarray = []
    def createLayers(self, nlayers, narray, ninput):
        res = []
        ## Este caso no estoy seguro
        if (nlayers==1):
            return res.append(NeuralLayer(narray[0], True, ninput))
        for i in range(0, nlayers):
            # Input Layer
            if (i == 0):
                l = NeuralLayer(narray[i], False, ninput)
            # Output Layer
            elif (i == nlayers-1):
                l = NeuralLayer(narray[i], True, narray[i-1])
            # Internal Layer
            else:
                l = NeuralLayer(narray[i], False, narray[i-1])
            res.append(l)
            i = i + 1
        self.layerarray = res
    def feed(self, input):
        output = input
        for l in self.layerarray:
            output = l.feed(output)
        return output
    def plotpoints(self):
        epoch = input('Ingrese la cantidad de épocas: ')
        epoch = int(epoch)
        trainingpoints = np.array([[0,0],[0,1],[1,0],[1,1]])
        while epoch > 0:
            for t in trainingpoints:
                if t[0] == t[1]: # XOR
                    self.train(t,0)
                else:
                    self.train(t,1)
            epoch = epoch - 1
        ## Clasificacion de los puntos reales
        newpoints = np.array([[0,0],[0,1],[1,0],[1,1]])
        listclassification = []
        for n in newpoints:
            if n[0] < n[1]: # Recta y = x
                listclassification.append(0)
            else:
                listclassification.append(1)
        output = []
        for po in range(0,len(newpoints)):
            classification = self.feed(newpoints[po])
            output.append(classification)
            if classification:
                plt.scatter(newpoints[po][0],newpoints[po][1],c='red')
            else:
                plt.scatter(newpoints[po][0],newpoints[po][1],c='blue')
        plt.plot([0,1], [0,1],c='black')
        aciertos = 0
        for l,o in zip(listclassification,output):
            aciertos += 1 if (l == o) else 0
        precision = aciertos / len(output)
        print(precision)
        plt.show()

class AndPerceptron(Perceptron):
    def __init__(self):
        Perceptron.__init__(self, np.array([1,0]), -0.5)
class OrPerceptron(Perceptron):
    def __init__(self):
        Perceptron.__init__(self, np.array([1,1]), -1.5)
class NandPerceptron(Perceptron):
    def __init__(self):
        Perceptron.__init__(self, np.array([-2,-2]), 3)
class SumPerceptron(Perceptron):
    def __init__(self):
        self.nand = NandPerceptron()
    def feed(self, x):
        res1 = self.nand.feed(x)
        res2 = self.nand.feed(np.array([x[0], res1]))
        res3 = self.nand.feed(np.array([x[1], res1]))
        ressum = self.nand.feed(np.array([res2, res3]))
        rescarry = self.nand.feed(np.array([res1,res1]))
        return np.array([rescarry,ressum])

class TrainPerceptron(Perceptron):
    def __init__(self):
        Perceptron.__init__(self, np.array([random.randint(-2,2),random.randint(-2,2)]),random.randint(-2,2),0.1)
class TrainSigmoid(Perceptron):
    def __init__(self):
        Sigmoid.__init__(self, np.array([random.randint(-2,2),random.randint(-2,2)]), random.randint(-2,2),0.1)

def main():
    choice = input('Seleccione operacion (AND=1, OR=2, NAND=3, SUM=4, TRAINPERCEPTRON=5, TRAINSIGMOID=6, PRECISIONPERCEPTRON=7, PRECISIONTRAINSIGMOID=8, NEURALNETWORK=9):')
    choice = int(choice)
    if (choice == 1):
        p = AndPerceptron()
    if (choice == 2):
        p = OrPerceptron()
    if (choice == 3):
        p = NandPerceptron()
    if (choice == 4):
        p = SumPerceptron()
    if (choice == 5):
        p = TrainPerceptron()
        p.plotpoints()
    if (choice == 6):
        p = TrainSigmoid()
        p.plotpoints()
    if (choice == 7):
        p = TrainPerceptron()
        p.plotlearning()
    if (choice == 8):
        p = TrainSigmoid()
        p.plotlearning()
    if (choice == 9):
        nn = NeuralNetwork(2, [2,1], 2)
        nn.createLayers(nn.nlayers, nn.narray, nn.ninput)
        nn.plotpoints()

if __name__ == "__main__":
    main()
