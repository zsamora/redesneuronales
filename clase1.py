import numpy as np
import unittest
import matplotlib.pyplot as plt
import random

class Perceptron:
    def __init__(self, w, b, lr=0):
        self.weight = w # Vector of weights
        self.bias = b   # Bias
        self.lrate = lr # Learning rate

    def feed(self, x):
        result = np.dot(self.weight, x) + self.bias
        if (result <= 0):
            return 0
        else:
            return 1

    def train(self, point, value):
        output = self.feed(point) # Perceptron's output according to w and b
        if (output != value):
            diff = value - output
            newweight = self.weight + (self.lr * point * diff) # Este se puede repetir en todos los input en vez de realizarlo solo una vez
            self.weight = newweight
            newbias = self.bias + (self.lr * diff)
            self.bias = newbias

    def plot(self, listpoint, listvalues, newpoints):
        for point,value in zip(listpoint,listvalues):
            self.train(point,value)


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
        Perceptron.__init__(self, np.array([random.randint(-2,2),random.randint(-2,2)]), random.randint(-2,2))

def main():
    choice = input('Seleccione operacion (AND=1, OR=2, NAND=3, SUM=4, TRAIN=5):')
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
        #trainingpoints = input('Ingrese valores: ')
        #trainingvalues = input('Ingrese clasificaciones: ')
        tpoints = input('Ingrese la cantidad de puntos para entrenar: ')
        tpoints = int(tpoints)
        npoints = input('Ingrese la cantidad de puntos a graficar: ')
        npoints = int(npoints)
        #trainingpoints = np.fromstring(trainingpoints, dtype=int, sep=' ')
        #print(trainingpoints)
        #trainingvalues = np.fromstring(trainingvalues, dtype=int, sep=' ')
        trainingpoints = np.random.rand(tpoints,2)
        trainingvalues = np.array([])
        for t in trainingpoints:
            if
            np.insert(trainingvalues, )
        newpoints = np.random.rand(npoints,2)
        p.plot(trainingpoints, trainingvalues, newpoints)
    x1 = input('Ingrese x1: ')
    x2 = input('Ingrese x2: ')
    result = p.feed(np.array([int(x1),int(x2)]))
    print(result)

if __name__ == "__main__":
    main()
