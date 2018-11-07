import unittest
from clase1 import *

class UnitTestClass(unittest.TestCase):
    def test_andPerceptron(self):
        andPerceptron = Perceptron(np.array([0.6 ,0.6]), -1) # (0.6*x0 + 0.6*x1) > 1
        self.assertFalse(andPerceptron.feed(np.array([0,0])))
        self.assertFalse(andPerceptron.feed(np.array([0,1])))
        self.assertFalse(andPerceptron.feed(np.array([1,0])))
        self.assertTrue(andPerceptron.feed(np.array([1,1])))
    def test_orPerceptron(self):
        orPerceptron = Perceptron(np.array([1 ,1]), -0.9)    # (1*x0 + 1*x1) > 0.9
        self.assertFalse(orPerceptron.feed(np.array([0,0])))
        self.assertTrue(orPerceptron.feed(np.array([0,1])))
        self.assertTrue(orPerceptron.feed(np.array([1,0])))
        self.assertTrue(orPerceptron.feed(np.array([1,1])))
    def test_nandPerceptron(self):
        nandPerceptron = Perceptron(np.array([-2 ,-2]), 3)   # (-2*x0 + -2*x1) > 3
        self.assertTrue(nandPerceptron.feed(np.array([0,0])))
        self.assertTrue(nandPerceptron.feed(np.array([0,1])))
        self.assertTrue(nandPerceptron.feed(np.array([1,0])))
        self.assertFalse(nandPerceptron.feed(np.array([1,1])))
    def test_sumPerceptron(self):
        sumPerceptron = SumPerceptron()
        self.assertTrue(np.array_equal(sumPerceptron.feed(np.array([0,0])),np.array([0,0])))
        self.assertTrue(np.array_equal(sumPerceptron.feed(np.array([0,1])),np.array([0,1])))
        self.assertTrue(np.array_equal(sumPerceptron.feed(np.array([1,0])),np.array([0,1])))
        self.assertTrue(np.array_equal(sumPerceptron.feed(np.array([1,1])),np.array([1,0])))

if __name__ == '__main__':
    p = Perceptron(np.array([random.randint(-2,2),random.randint(-2,2)]),random.randint(-2,2), 0.1)
    print("Test para clasificación de recta y = x con Perceptron \n(Recuerde cerrar el gráfico para continuar el testing)")
    tpoints = input('Ingrese la cantidad de puntos para entrenar: ')
    tpoints = int(tpoints)
    npoints = input('Ingrese la cantidad de puntos a graficar: ')
    npoints = int(npoints)
    p.plotpoints(tpoints,npoints)
    newp = Perceptron(np.array([random.randint(-2,2),random.randint(-2,2)]),random.randint(-2,2), 0.1)
    print("Curva de precisión vs n° de aprendizajes del Perceptron \n(Recuerde cerrar el gráfico para continuar el testing)")
    npoints = input('Ingrese la cantidad de puntos a graficar: ')
    npoints = int(npoints)
    ntrain = input('Ingrese el número de entrenamientos: ')
    ntrain = int(ntrain)
    newp.plotlearning(npoints, ntrain)
    s = Sigmoid(np.array([random.randint(-2,2),random.randint(-2,2)]), random.randint(-2,2), 0.1)
    print("Test para clasificación de recta y = x con Sigmoid \n(Recuerde cerrar el gráfico para continuar el testing)")
    tpoints = input('Ingrese la cantidad de puntos para entrenar: ')
    tpoints = int(tpoints)
    npoints = input('Ingrese la cantidad de puntos a graficar: ')
    npoints = int(npoints)
    s.plotpoints(tpoints,npoints)
    news = Sigmoid(np.array([random.randint(-2,2),random.randint(-2,2)]), random.randint(-2,2), 0.1)
    print("Curva de precisión vs n° de aprendizajes del Sigmoid \n(Recuerde cerrar el gráfico para continuar el testing)")
    npoints = input('Ingrese la cantidad de puntos a graficar: ')
    npoints = int(npoints)
    ntrain = input('Ingrese el número de entrenamientos: ')
    ntrain = int(ntrain)
    news.plotlearning(npoints, ntrain)
    unittest.main()
