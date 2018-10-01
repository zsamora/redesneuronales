import numpy as np
import unittest

class Perceptron:
    def __init__(self, w, b):
        self.weight = w # Vector of weights
        self.bias = b   # Bias

    def feed(self, x):
        result = np.dot(self.weight, x) + self.bias
        if (result <= 0):
            return 0
        else:
            return 1

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

def main():
    choice = input('Seleccione operacion (AND=1, OR=2, NAND=3, SUM=4):')
    choice = int(choice)
    if (choice == 1):
        p = AndPerceptron()
    if (choice == 2):
        p = OrPerceptron()
    if (choice == 3):
        p = NandPerceptron()
    if (choice == 4):
        p = SumPerceptron()
    x1 = input('Ingrese x1: ')
    x2 = input('Ingrese x2: ')
    result = p.feed(np.array([int(x1),int(x2)]))
    print(result)

if __name__ == "__main__":
    main()
