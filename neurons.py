import numpy as np
import unittest
import matplotlib.pyplot as plt
import random
import math

class ArtificialNeuron:
    def __init__(self, w, b, lr=0.1, precision = 0, output = 0, delta = 0):
        self.weight = w # Vector of weights
        self.bias = b   # Bias
        self.lr = lr    # Learning rate
        self.precision = precision
        self.output = output
        self.delta = delta

    def train(self, point, expected):
        real = self.feed(point)
        if (expected != real):
            diff = expected - real
            self.weight = self.weight + (self.lr * point * diff)
            self.bias = self.bias + (self.lr * diff)

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
        if (result <= 0):
            return 0
        else:
            return 1

class Sigmoid(ArtificialNeuron):
    def feed(self, x):
        #print("input:",x,"weight:",self.weight)
        result = 1.0 / (1.0 + math.exp(-1.0 * (np.dot(self.weight, x) + self.bias)))
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
    def __init__(self, type=1, nneurons=1, isoutput=True, ninput=0, lr=0.5):
        self.isoutput = isoutput
        self.neuronarray = []
        while (nneurons > 0):
            if type:
                n = Sigmoid(np.random.rand(ninput)*2, random.randint(-2,2), lr)
            else:
                n = Perceptron(np.random.rand(ninput)*2, random.randint(-2,2), lr)
            self.neuronarray.append(n)
            nneurons = nneurons - 1

    def feed (self, input):
        output = []
        for n in self.neuronarray:
            output.append(n.feed(input))
        return output

class NeuralNetwork():
    def __init__(self, type, nlayers=1, narray=[1], ninput=1, lr=0.5, error=0):
        self.layerarray = []
        # Solo una capa de neuronas
        if (nlayers==1):
            self.layerarray.append(NeuronLayer(type, narray[0], True, ninput, lr))
        else:
            for i in range(0, nlayers):
                if (i == 0): # Input Layer
                    l = NeuronLayer(type, narray[i], False, ninput, lr)
                elif (i == nlayers-1): # Output Layer
                    l = NeuronLayer(type, narray[i], True, narray[i-1], lr)
                else: # Internal Layer
                    l = NeuronLayer(type, narray[i], False, narray[i-1], lr)
                self.layerarray.append(l)

    def feed(self, input):
        output = input
        for l in self.layerarray:
            output = l.feed(output)
        return output

    def train(self, input, expected, epochs):
        while (epochs > 0):
            for i in range(0,len(input)):
                output = np.array(self.feed(input[i]))
                error = expected[i] - output
                delta = error * output * (1 - output)
                for j in range(len(self.layerarray[-1].neuronarray)):
                    self.layerarray[-1].neuronarray[j].delta = delta[j]
                self.backwardPropagation()
                self.updateWeigths(input[i])
            epochs = epochs - 1

    def backwardPropagation(self):
        for j in reversed(range(0,len(self.layerarray)-1)):
            # Neuronas en capa J (de mas externa a interna) de las hidden layers
            for n in range(0, len(self.layerarray[j].neuronarray)):
                error = 0
                for m in self.layerarray[j+1].neuronarray:
                    error = error + m.weight[n] * m.delta
                self.layerarray[j].neuronarray[n].error = error
                self.layerarray[j].neuronarray[n].delta = error * self.layerarray[j].neuronarray[n].output * (1 - self.layerarray[j].neuronarray[n].output)

    def updateWeigths(self, input):
        output = input
        for l in self.layerarray:
            outputnew = []
            for i in range(0,len(l.neuronarray)):
                for k in range(0,len(l.neuronarray[i].weight)):
                    l.neuronarray[i].weight[k] = l.neuronarray[i].weight[k] + (l.neuronarray[i].lr * l.neuronarray[i].delta * output[k])
                l.neuronarray[i].bias = l.neuronarray[i].bias + (l.neuronarray[i].lr * l.neuronarray[i].delta)
                outputnew.append(l.neuronarray[i].output)
            output = outputnew

    def setwb(self, layer, neuron, w, b):
        self.layerarray[layer].neuronarray[neuron].weight = w
        self.layerarray[layer].neuronarray[neuron].bias = b

    def getwb(self, layer, neuron):
        return [self.layerarray[layer].neuronarray[neuron].weight, self.layerarray[layer].neuronarray[neuron].bias]

    def getlr(self, layer, neuron):
        return self.layerarray[layer].neuronarray[neuron].lr

    def checkw(self, w):
        for l in self.layerarray:
            for n in l.neuronarray:
                for p in range(len(n.weight)):
                    while (abs(n.weight[p]) < w):
                        n.weight[p] = random.randint(-2,2)

    def plottrain(self, input, expected, epochs):
        e = []
        for i in range (0, epochs):
            errorsq = 0
            for i in range(0,len(input)):
                output = np.array(self.feed(input[i]))
                error = expected[i] - output
                errorsq = errorsq + sum(map(lambda x:x*x,error))
                delta = error * output * (1 - output)
                for j in range(len(self.layerarray[-1].neuronarray)):
                    self.layerarray[-1].neuronarray[j].delta = delta[j]
                self.backwardPropagation()
                self.updateWeigths(input[i])
            e.append(errorsq)
        plt.plot(range(0,epochs), e)
        plt.show()
