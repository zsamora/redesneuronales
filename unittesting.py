import unittest
import csv
from sklearn.model_selection import train_test_split
from neurons import *
PERCEPTRON = 0
SIGMOID = 1

class UnitTestClass(unittest.TestCase):
    def test_andPerceptron(self):
        andPerceptron = Perceptron(np.array([0.6 ,0.6]), -1)
        self.assertFalse(andPerceptron.feed(np.array([0,0])))
        self.assertFalse(andPerceptron.feed(np.array([0,1])))
        self.assertFalse(andPerceptron.feed(np.array([1,0])))
        self.assertTrue(andPerceptron.feed(np.array([1,1])))
    def test_orPerceptron(self):
        orPerceptron = Perceptron(np.array([1 ,1]), -0.9)
        self.assertFalse(orPerceptron.feed(np.array([0,0])))
        self.assertTrue(orPerceptron.feed(np.array([0,1])))
        self.assertTrue(orPerceptron.feed(np.array([1,0])))
        self.assertTrue(orPerceptron.feed(np.array([1,1])))
    def test_nandPerceptron(self):
        nandPerceptron = Perceptron(np.array([-2 ,-2]), 3)
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
    def test_xorNeuralNetwork(self):
        xorNeuralNetwork = NeuralNetwork(PERCEPTRON, 2, np.array([2,1]), 2)
        xorNeuralNetwork.setwb(0,0,np.array([0.6, 0.6]),-1)
        xorNeuralNetwork.setwb(0,1,np.array([1.1, 1.1]),-1)
        xorNeuralNetwork.setwb(1,0,np.array([-2, 1.1]),-1)
        self.assertFalse(xorNeuralNetwork.feed(np.array([0, 0]))[0])
        self.assertTrue(xorNeuralNetwork.feed(np.array([0, 1]))[0])
        self.assertTrue(xorNeuralNetwork.feed(np.array([1, 0]))[0])
        self.assertFalse(xorNeuralNetwork.feed(np.array([1, 1]))[0])
    def test_ex1NeuralNetwork(self):
        ex1NeuralNetwork = NeuralNetwork(SIGMOID, 2, np.array([1,1]), 2)
        ex1NeuralNetwork.setwb(0, 0, np.array([0.4, 0.3]), 0.5)
        ex1NeuralNetwork.setwb(1, 0, np.array([0.3]), 0.4)
        ex1NeuralNetwork.train(np.array([np.array([1,1])]), np.array([1]), 1)
        self.assertTrue(math.isclose(ex1NeuralNetwork.getwb(0, 0)[1], 0.502101508999489))
        self.assertTrue(math.isclose(ex1NeuralNetwork.getwb(0, 0)[0][0],0.40210150899948904))
        self.assertTrue(math.isclose(ex1NeuralNetwork.getwb(0, 0)[0][1],0.302101508999489))
        self.assertTrue(math.isclose(ex1NeuralNetwork.getwb(1, 0)[1], 0.43937745312797394))
        self.assertTrue(math.isclose(ex1NeuralNetwork.getwb(1, 0)[0][0], 0.33026254863991883))
    def test_ex2NeuralNetwork(self):
        ex1NeuralNetwork = NeuralNetwork(SIGMOID, 2, np.array([2,2]), 2)
        ex1NeuralNetwork.setwb(0, 0, np.array([0.7, 0.3]), 0.5)
        ex1NeuralNetwork.setwb(0, 1, np.array([0.3, 0.7]), 0.4)
        ex1NeuralNetwork.setwb(1, 0, np.array([0.2, 0.3]), 0.3)
        ex1NeuralNetwork.setwb(1, 1, np.array([0.4, 0.2]), 0.6)
        ex1NeuralNetwork.train(np.array([np.array([1,1])]), np.array(np.array([1,1])), 1)
        self.assertTrue(math.isclose(ex1NeuralNetwork.getwb(0, 0)[1], 0.5025104485493278))
        self.assertTrue(math.isclose(ex1NeuralNetwork.getwb(0, 0)[0][0], 0.7025104485493278))
        self.assertTrue(math.isclose(ex1NeuralNetwork.getwb(0, 0)[0][1], 0.3025104485493278))
        self.assertTrue(math.isclose(ex1NeuralNetwork.getwb(0, 1)[1], 0.40249801135748337))
        self.assertTrue(math.isclose(ex1NeuralNetwork.getwb(0, 1)[0][0],0.30249801135748333))
        self.assertTrue(math.isclose(ex1NeuralNetwork.getwb(0, 1)[0][1],0.7024980113574834))
        self.assertTrue(math.isclose(ex1NeuralNetwork.getwb(1, 0)[1], 0.3366295422515899))
        self.assertTrue(math.isclose(ex1NeuralNetwork.getwb(1, 0)[0][0],0.22994737881955657))
        self.assertTrue(math.isclose(ex1NeuralNetwork.getwb(1, 0)[0][1],0.32938362863950127))
        self.assertTrue(math.isclose(ex1NeuralNetwork.getwb(1, 1)[1], 0.6237654881509048))
        self.assertTrue(math.isclose(ex1NeuralNetwork.getwb(1, 1)[0][0],0.41943005652646226))
        self.assertTrue(math.isclose(ex1NeuralNetwork.getwb(1, 1)[0][1],0.21906429169838573))
    def test_xorTrainedNeuralNetwork(self):
        xorTrainedNeuralNetwork = NeuralNetwork(SIGMOID, 2, np.array([2,1]), 2)
        xorTrainedNeuralNetwork.checkw(0.3)
        xorTrainedNeuralNetwork.train(np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([[0],[1],[1],[0]]), 150000)
        output00 = xorTrainedNeuralNetwork.feed(np.array([0, 0]))[0]
        output01 = xorTrainedNeuralNetwork.feed(np.array([0, 1]))[0]
        output10 = xorTrainedNeuralNetwork.feed(np.array([1, 0]))[0]
        output11 = xorTrainedNeuralNetwork.feed(np.array([1, 1]))[0]
        print("\nSi este test falla, deben observarse los siguientes valores correspondientes")
        print("a los outputs obtenidos,para ver la diferencia que tienen a 1 o 0 (el rango actual de tolerancia es de 0.005):")
        print("\nOutputs: \n- 00:", output00, "\n- 01:", output01, "\n- 10:", output10, "\n- 11:", output11)
        self.assertFalse(math.isclose(0,output00,rel_tol=5e-03))
        self.assertTrue(math.isclose(1,output01,rel_tol=5e-03))
        self.assertTrue(math.isclose(1,output10,rel_tol=5e-03))
        self.assertclassesFalse(math.isclose(0,output11,rel_tol=5e-03))
    def test_datasetNeuralNetwork(self):
        print("\nEn este test se pone a prueba la precisión de la predicción, se solicita el ingreso de la cantidad de filas a trabajar del dataset (máximo 1599) y la cantidad de épocas de entrenamiento, por temas de tiempo de ejecución se recomienda mantener una relación aproximada de (filas*epocas = 10000000)")
        filas = input('\nIngrese la cantidad de filas del dataset con las cuales trabajar: ')
        filas = int(filas)
        epocas = input('Ingrese la cantidad de épocas para entrenar: ')
        epocas = int(epocas)
        # Red is class [1,0]
        reddata = np.genfromtxt("winequality-red.csv",  delimiter=';')
        reddata = reddata[1:filas]
        redclass = np.array([np.array([1,0]) for i in range(len(reddata))])
        # White is class [0,1]
        whitedata = np.genfromtxt("winequality-white.csv",  delimiter=';')
        whitedata = whitedata[1:filas]
        whiteclass = np.array([np.array([0,1]) for i in range(len(whitedata))])
        data = np.concatenate((reddata, whitedata), axis=0)
        # Delete quality column
        data = np.delete(data, 11, 1)
        classes = np.concatenate((redclass, whiteclass), axis=0)
        # Normalization
        datanorm = data / data.max(axis=0)
        # Randomize, only for simplicity
        X_train, X_test, y_train, y_test = train_test_split(datanorm, classes, test_size=.3, random_state=37, stratify=classes)
        datasetNeuralNetwork = NeuralNetwork(SIGMOID, 3, np.array([2,3,2]), 11)
        datasetNeuralNetwork.checkw(0.3)
        datasetNeuralNetwork.train(X_train, y_train, epocas)
        TP = 0.0
        FP = 0.0
        for i in range(len(X_test)):
            output = datasetNeuralNetwork.feed(X_test[i])
            if (y_test[i][output.index(max(output))] and math.isclose(max(output),1,rel_tol=0.3)):
                TP = TP + 1.0
            else:
                FP = FP + 1.0
        print("Precision con", epocas, "epocas de entrenamiento y", filas, "filas:", TP / (TP + FP))
    def test_hlayersNeuralNetwork(self):
        print("\nEn este test se solicita el ingreso de la cantidad de capas ocultas y un número de neuronas por capa, se recomienda por tiempo de ejecución que capas*neuronas < 30 (Recuerde cerrar el gráfico)")
        capas = input('\nIngrese la cantidad de capas: ')
        capas = int(capas)
        nneu = input('Ingrese la cantidad de neuronas por capa: ')
        nneu = int(nneu)
        # Red is class [1,0]
        reddata = np.genfromtxt("winequality-red.csv",  delimiter=';')
        reddata = reddata[1:101]
        redclass = np.array([np.array([1,0]) for i in range(len(reddata))])
        # White is class [0,1]
        whitedata = np.genfromtxt("winequality-white.csv",  delimiter=';')
        whitedata = whitedata[1:101]
        whiteclass = np.array([np.array([0,1]) for i in range(len(whitedata))])
        data = np.concatenate((reddata, whitedata), axis=0)
        # Delete quality column
        data = np.delete(data, 11, 1)
        classes = np.concatenate((redclass, whiteclass), axis=0)
        # Normalization
        datanorm = data / data.max(axis=0)
        # Randomize, only for simplicity
        X_train, X_test, y_train, y_test = train_test_split(datanorm, classes, test_size=.3, random_state=37, stratify=classes)
        datasetNeuralNetwork = NeuralNetwork(SIGMOID, capas, np.append(np.array([nneu]*(capas-1)),np.array([2])), 11)
        datasetNeuralNetwork.checkw(0.3)
        datasetNeuralNetwork.plottrain(X_train, y_train, 1000)
        datasetNeuralNetwork = NeuralNetwork(SIGMOID, capas, np.append(np.array([nneu]*(capas-1)),np.array([2])), 11)
        datasetNeuralNetwork.checkw(0.3)
        datasetNeuralNetwork.train(X_train, y_train, 1000)
        TP = 0.0
        FP = 0.0
        for i in range(len(X_test)):
            output = datasetNeuralNetwork.feed(X_test[i])
            if (y_test[i][output.index(max(output))] and math.isclose(max(output),1,rel_tol=0.3)):
                TP = TP + 1.0
            else:
                FP = FP + 1.0
        print("Precision con", capas, "capas de", nneu,"neuronas cada una:", TP / (TP + FP))
    def test_lrNeuralNetwork(self):
        print("\nEn este test, se solicita el ingreso del learning rate (valor entre 0 y 1)")
        newlr = input('\nIngrese el learning rate: ')
        newlr = float(newlr)
        # Red is class [1,0]
        reddata = np.genfromtxt("winequality-red.csv",  delimiter=';')
        reddata = reddata[1:101]
        redclass = np.array([np.array([1,0]) for i in range(len(reddata))])
        # White is class [0,1]
        whitedata = np.genfromtxt("winequality-white.csv",  delimiter=';')
        whitedata = whitedata[1:101]
        whiteclass = np.array([np.array([0,1]) for i in range(len(whitedata))])
        data = np.concatenate((reddata, whitedata), axis=0)
        # Delete quality column
        data = np.delete(data, 11, 1)
        classes = np.concatenate((redclass, whiteclass), axis=0)
        # Normalization
        datanorm = data / data.max(axis=0)
        # Randomize, only for simplicity
        X_train, X_test, y_train, y_test = train_test_split(datanorm, classes, test_size=.3, random_state=37, stratify=classes)
        datasetNeuralNetwork = NeuralNetwork(SIGMOID, 3, np.array([2,3,2]), 11, newlr)
        datasetNeuralNetwork.checkw(0.3)
        datasetNeuralNetwork.plottrain(X_train, y_train, 1000)
        datasetNeuralNetwork = NeuralNetwork(SIGMOID, 3, np.array([2,3,2]), 11, newlr)
        datasetNeuralNetwork.checkw(0.3)
        datasetNeuralNetwork.train(X_train, y_train, 1000)
        TP = 0.0
        FP = 0.0
        for i in range(len(X_test)):
            output = datasetNeuralNetwork.feed(X_test[i])
            if (y_test[i][output.index(max(output))] and math.isclose(max(output),1,rel_tol=0.3)):
                TP = TP + 1.0
            else:
                FP = FP + 1.0
        print("Precision con lr =", newlr,":", TP / (TP + FP))

if __name__ == '__main__':
    p = Perceptron(np.array([random.randint(-2,2),random.randint(-2,2)]),random.randint(-2,2), 0.1)
    print("Test para clasificación de recta y = x con Perceptron \n(Recuerde cerrar el gráfico para continuar el testing)")
    tpoints = input('Ingrese la cantidad de puntos para entrenar: ')
    tpoints = int(tpoints)
    npoints = input('Ingrese la cantidad de puntos a graficar: ')
    npoints = int(npoints)
    p.plotpoints(tpoints,npoints)
    newp = Perceptron(np.array([random.randint(-2,2),random.randint(-2,2)]),random.randint(-2,2), 0.1)
    print("\nCurva de precisión vs n° de aprendizajes del Perceptron \n(Recuerde cerrar el gráfico para continuar el testing)")
    npoints = input('Ingrese la cantidad de puntos a graficar: ')
    npoints = int(npoints)
    ntrain = input('Ingrese el número de entrenamientos: ')
    ntrain = int(ntrain)
    newp.plotlearning(npoints, ntrain)
    s = Sigmoid(np.array([random.randint(-2,2),random.randint(-2,2)]), random.randint(-2,2), 0.1)
    print("\nTest para clasificación de recta y = x con Sigmoid \n(Recuerde cerrar el gráfico para continuar el testing)")
    tpoints = input('Ingrese la cantidad de puntos para entrenar: ')
    tpoints = int(tpoints)
    npoints = input('Ingrese la cantidad de puntos a graficar: ')
    npoints = int(npoints)
    s.plotpoints(tpoints,npoints)
    news = Sigmoid(np.array([random.randint(-2,2),random.randint(-2,2)]), random.randint(-2,2), 0.1)
    print("\nCurva de precisión vs n° de aprendizajes del Sigmoid \n(Recuerde cerrar el gráfico para continuar el testing)")
    npoints = input('Ingrese la cantidad de puntos a graficar: ')
    npoints = int(npoints)
    ntrain = input('Ingrese el número de entrenamientos: ')
    ntrain = int(ntrain)
    news.plotlearning(npoints, ntrain)
    print("\nCurva de error vs n° de epocas de XOR en la Neural Network \n(Recuerde cerrar el gráfico para continuar el testing)")
    epochs = input('Ingrese la cantidad de epocas a entrenar: ')
    epochs = int(epochs)
    xorTrainedNeuralNetwork = NeuralNetwork(SIGMOID, 2, np.array([2,1]), 2)
    xorTrainedNeuralNetwork.plottrain(np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([[0],[1],[1],[0]]), epochs)
    print("\nEl último test toma algunos segundos, espere...\n")
    unittest.main()
