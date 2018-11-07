import numpy as np
import unittest
import matplotlib.pyplot as plt
import random
import math

class ArtificialNeuron:
    def __init__(self, w, b, lr=0.1, precision = 0, output = 0, delta = 0):
        self.weight = w # Vector of weights
        self.bias = b   # Bias
        self.lr = lr # Learning rate
        self.precision = precision
        self.output = output
        self.delta = delta
        ## Clase 22/10 Learning rate = 0.5
    def train(self, point, expected):
        real = self.feed(point) # Perceptron's output according to w and b
        if (expected != real):
            diff = expected - real
            newweight = self.weight + (self.lr * point * diff) # Este se puede repetir en todos los input en vez de realizarlo solo una vez)
            self.weight = newweight
            newbias = self.bias + (self.lr * diff)
            self.bias = newbias

    def plotpoints(self, tpoints, npoints):
        trainingpoints = np.random.rand(tpoints,2)
        for t in trainingpoints:
            # Recta y = x
            if t[0] < t[1]:
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
            classification = classification > 0.5 # Treshold para el Sigmoid (Perceptron queda igual)
            output.append(classification)
            if classification:
                plt.scatter(newpoints[po][0],newpoints[po][1],c='red')
            else:
                plt.scatter(newpoints[po][0],newpoints[po][1],c='blue')
        plt.plot([0,1], [0,1],c='black')
        plt.show()

    def plotlearning(self, npoints, ntrain):
        p = []
        for i in range (0,ntrain):
            p.append(self.precisionlearning(i,npoints))
        plt.plot(range(0,ntrain),p)
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
            classification = classification > 0.5 # Treshold para el Sigmoid (Perceptron queda igual)
            output.append(classification)
        aciertos = 0
        for l,o in zip(listclassification,output):
            aciertos += 1 if (l == o) else 0
        precision = aciertos / len(output)
        return precision

class Perceptron(ArtificialNeuron):
    def feed(self, x):
        result = np.dot(self.weight, x) + self.bias
        self.output = result
        if (result <= 0):
            return 0
        else:
            return 1

class Sigmoid(ArtificialNeuron):
    def feed(self, x):
        result = 1 / (1 + math.exp(-1 * np.dot(self.weight, x) - self.bias))
        self.output = result
        return result

class SumPerceptron(ArtificialNeuron):
    def __init__(self):
        self.nand = Perceptron(np.array([-2 ,-2]), 3) # Nand Perceptron
    def feed(self, x):
        res1 = self.nand.feed(x)
        res2 = self.nand.feed(np.array([x[0], res1]))
        res3 = self.nand.feed(np.array([x[1], res1]))
        ressum = self.nand.feed(np.array([res2, res3]))
        rescarry = self.nand.feed(np.array([res1,res1]))
        return np.array([rescarry,ressum])

class NeuronLayer():
    def __init__(self, nneurons=1, isoutput=True, ninput=0):
        self.isoutput = isoutput
        self.neuronarray = []
        while (nneurons > 0):
            n = Sigmoid(np.random.rand(ninput)*2, random.randint(-2,2), 0.1)
            self.neuronarray.append(n)
            nneurons = nneurons - 1

    def feed (self, input):
        output = []
        for n in self.neuronarray:
            output.append(n.feed(input))
        return output

class NeuralNetwork():
    def __init__(self, nlayers=1, narray=[1], ninput=1):
        self.layerarray = []
        if (nlayers==1):
            self.layerarray.append(NeuronLayer(narray[0], True, ninput))
        else:
            for i in range(0, nlayers):
                if (i == 0): # Input Layer
                    l = NeuronLayer(narray[i], False, ninput)
                elif (i == nlayers-1): # Output Layer
                    l = NeuronLayer(narray[i], True, narray[i-1])
                else: # Internal Layer
                    l = NeuronLayer(narray[i], False, narray[i-1])
                self.layerarray.append(l)

    def feed(self, input):
        output = input
        for l in self.layerarray:
            output = l.feed(output)
        return output

    def train(self, input, expected):
        output = self.feed(input) # Final output
        error = expected - output[0]
        delta = error * output[0] * (1 - output[0])
        # TODO: Asumo que es una neurona de salida por ahora
        self.layerarray[len(self.layerarray)-1].neuronarray[0].delta = delta
        for j in reversed(range(0,len(self.layerarray)-2):
            error = 0
            for n in self.layerarray[j+1].neuronarray:
                error = error + np.dot(n.weight,n.delta)
            for m in self.layerarray[j].neuronarray:
                m.delta = error * m.output * (1 - m.output)
        self.updateWeigths(input)
        return error

    def updateWeigths(self, input):
        return

    def setwb(self, layer, neuron, w, b):
        self.layerarray[layer].neuronarray[neuron].weight = w
        self.layerarray[layer].neuronarray[neuron].bias = b

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
