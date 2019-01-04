import operator
import random
import copy
import math

function_set = {"+" : operator.add,
                "-" : operator.sub,
                "*" : operator.mul,
                "/" : operator.truediv}

# Hiperparámetros
MAX_DEPTH = 6
NN_AST = (2**(MAX_DEPTH+1) - 1)
POP_SIZE = 500
TOUR_SIZE = int((POP_SIZE * 3) /  4)   # Size of tournament
N_GEN = 0
N_NUMBERS = 10
MIN_VALUE = 0
MAX_VALUE = 100
MIN_SOL = 0
MAX_SOL = 1000
MAX_GEN = 200
TOL = 5
TOL_EQ = 30
MUT_RATE = 0.01
CROSS_RATE = 0.9

class AST:
    def __init__(self, numbers_set, has_vars = False, vars = None):
        self.root = None
        self.treelist = [0]*NN_AST
        self.nset = numbers_set
        self.path = []
        self.has_vars = has_vars
        self.vars = vars

    def calculate(self, values = None):
        if values != None:
            dict = {}
            for i,j in zip(self.vars, values):
                dict[i] = j
            return self.root.calculate(dict)
        return self.root.calculate()

    def __str__(self):
        return self.root.__str__()

    def initialize(self, depth, index):
        self.treelist[index] = 1
        depth += 1
        # Favorecer la creación de funciones por sobre terminal (opcional)
        is_fun = random.random() < 0.5
        if index == 0:
            self.root = Function(self.initialize(depth, 2*index+1), self.initialize(depth, 2*index+2),random.choice(list(function_set.keys())))
            return self
        elif depth == MAX_DEPTH or not(is_fun):
            if self.has_vars:
                if random.random() < 0.5:
                    return Terminal(random.choice(self.nset))
                else:
                    return Terminal(random.choice(self.vars))
            else:
                return Terminal(random.choice(self.nset))
        elif is_fun:
            return Function(self.initialize(depth, 2*index+1), self.initialize(depth, 2*index+2),random.choice(list(function_set.keys())))

    def subtree(self):
        self.path = []
        actualindex = 0
        mixpoint = random.randint(1, NN_AST-1)
        # Iterar hasta escoger un nodo valido
        while not(self.treelist[mixpoint]):
            mixpoint = random.randint(1,NN_AST-1)
        index = mixpoint
        # Si el nodo seleccionado es distinto de la raiz, crear camino
        while mixpoint != 0:
            self.path.insert(0,mixpoint)
            mixpoint = int((mixpoint-1)/2)
        # Si esta vacio, retornar la raiz
        if self.path == []:
            return self.root
        # Si no, búsqueda recursiva
        else:
            #print("index", index, "- path", self.path)
            return self.root.subtree(index, self.path, actualindex)

    def crossover(self, subtree):
        self.path = []
        actualindex = 0
        mixpoint = random.randint(1, NN_AST-1)
        # Iterar hasta escoger un nodo valido
        while not(self.treelist[mixpoint]):
            mixpoint = random.randint(1,NN_AST-1)
        index = mixpoint
        # Si el nodo seleccionado es distinto de la raiz, crear camino
        while mixpoint != 0:
            self.path.insert(0,mixpoint)
            mixpoint = int((mixpoint-1)/2)
        #print(index, self.path)
        # Si esta vacio, retornar la raiz
        if self.path == []:
            self.root = subtree
        # Si no, búsqueda recursiva
        else:
            self.root.crossover(index, subtree, self.path, actualindex)

    def mutate(self):
        if self.has_vars:
            self.crossover(AST(numbers_set, True, vars).initialize(0,0).root)
        else:
            self.crossover(AST(self.nset).initialize(0,0).root)

class Function:
    def __init__(self, left, right, op):
        self.left = left
        self.right = right
        self.op = op

    def calculate(self, dict = None):
        return function_set[self.op](self.left.calculate(dict), self.right.calculate(dict))

    def __str__(self):
        return "(" + self.left.__str__() + " " + self.op + " " + self.right.__str__() + ")"

    def subtree(self, index, path, actualindex):
        leftindex = (2*actualindex + 1)
        rightindex = (2*actualindex + 2)
        #print("index", self.index, "sl", leftindex, "sr", rightindex,"indexbuscado", index,"path", path)
        # Indice hijo izquierdo es el buscado
        if (leftindex == index):
            #print("izquierdo")
            return self.left
        # Indice hijo derecho es el buscado
        elif (rightindex == index):
            #print("derecho")
            return self.right
        # Hijo izquierdo es el camino
        elif (leftindex == path[0]):
            #print("seleccionado nodo izquierdo")
            return self.left.subtree(index, path[1:], leftindex)
        # Hijo derecho es el camino
        elif (rightindex == path[0]):
            #print("seleccionado nodo derecho")
            return self.right.subtree(index, path[1:], rightindex)
        else:
            #print("index", index, "|| path", path, "|| actualindex", actualindex, "|| lindex", leftindex, "|| rindex", rightindex)
            #print("Nodo inválido seleccionado en subtree Function")
            return False

    def crossover(self, index, subtree, path, actualindex):
        #print("index", self.index, "sl", self.left.index, "sr", self.right.index,"indexbuscado", index,"path", path)
        leftindex = (2*actualindex + 1)
        rightindex = (2*actualindex + 2)
        # Indice hijo izquierdo es el buscado
        if (leftindex == index):
            self.left = subtree
        # Indice hijo derecho es el buscado
        elif (rightindex == index):
            self.right = subtree
        # Hijo izquierdo es el camino
        elif (leftindex == path[0]):
            return self.left.crossover(index, subtree, path[1:], leftindex)
        # Hijo derecho es el camino
        elif (rightindex == path[0]):
            return self.right.crossover(index, subtree, path[1:], rightindex)
        else:
            #print("index", index, "|| path", path, "|| actualindex", actualindex, "|| lindex", leftindex, "|| rindex", rightindex)
            #print("Nodo inválido seleccionado en crossover Function")
            return False

class Terminal:
    def __init__(self, val):
        self.val = val

    def calculate(self, dict):
        if type(self.val) == str:
            return dict[self.val]
        return self.val

    def __str__(self):
        return str(self.val)

    def subtree(self, index, path, actualindex):
        #print("index", index, "|| path", path, "|| actualindex", actualindex)
        #print("Nodo inválido seleccionado en subtree Terminal")
        return False

    def crossover(self, index, subtree, path, actualindex):
        #print("index", index, "|| path", path, "|| actualindex", actualindex)
        #print("Nodo inválido seleccionado en crossover Terminal")
        return False

def createNumberSet():
    numbers_set = []
    while len(numbers_set) != N_NUMBERS:
        numbers_set.append(random.randint(MIN_VALUE,MAX_VALUE))
    return numbers_set

def createPopulationNumbers(numbers_set, has_vars = False, vars = None):
    population = []
    while len(population) != POP_SIZE:
        equation = AST(numbers_set, has_vars, vars).initialize(0,0)
        try:
            if not(has_vars):
                equation.calculate()
            population.append(equation)
        except:
            continue
    return population

def fitnessNumbers(individual, solution):
    initiate = False
    if individual.has_vars:
        result = math.inf
        for i in range(-10,10):
            try:
                aux = abs(solution.calculate([i]) - individual.calculate([i]))
                if not(initiate):
                    result = 0
                    initiate = True
                result += aux
            except Exception as e:
                #print(e)
                continue
        return result
    return abs(solution - individual.calculate())

def tournamentSelectionNumbers(population, fitness):
    best = None
    fit = 0
    for i in range(TOUR_SIZE):
        index = random.randint(0,POP_SIZE-1)
        if (best == None or fitness[index] < fitness[best]):
            fit = fitness[index]
            best = index
    #print("Fitness:",fit)
    return population[best]

def reproductionNumbers(parentsub, parentcross):
    new_population = []
    #print("###############")
    while len(new_population) != POP_SIZE:
        subtree = False
        # Mientras el subtree sea falso (o inválido)
        while (subtree == False):
            subtree = copy.deepcopy(parentsub.subtree())
            #print("subtree",subtree)
            try:
                if not(parentsub.has_vars):
                    subtree.calculate()
            except:
                subtree = False
                continue
        child = False
        while (child == False):
            paux = copy.deepcopy(parentcross)
            #print("paux", paux)
            if random.random() < CROSS_RATE:
                paux.crossover(subtree)
                #print("cros", paux)
            try:
                if not(paux.has_vars):
                    paux.calculate()
                if random.random() < MUT_RATE:
                    #print("Mutation")
                    paux.mutate()
                    if not(paux.has_vars):
                        paux.calculate()
                child = paux
            except:
                continue
        new_population.append(child)
        #print("###############")
    return new_population

def geneticProgrammingNumbers(population, solution):
    global N_GEN
    while N_GEN < MAX_GEN:
        fitness = []
        for p in population:
            f = round(fitnessNumbers(p, solution),2)
            if p.has_vars:
                if f <= TOL_EQ:
                    return p, N_GEN
            else:
                if f <= TOL:
                    return p, N_GEN
            fitness.append(f)
        #print("\nFitness:", fitness)
        #print("\n#############################")
        #mean = sum(fitness) / len(fitness)
        #fitmean.append(mean)
        #variance.append(sum((fit - mean) ** 2 for fit in fitness) / len(fitness))
        parent1 = tournamentSelectionNumbers(population, fitness)
        parent2 = tournamentSelectionNumbers(population, fitness)
        #print("\nTournament parents:")
        #print(parent1)
        #print(parent2)
        # Parent 1 se le extrae un sub-árbol
        if random.random() < 0.5:
            population = reproductionNumbers(parent1, parent2)
        # Parent 2 se le extrae un sub-árbol
        else:
            population = reproductionNumbers(parent2, parent1)
        #print("\nNew population:")
        #for p in population:
        #    print(p)
        N_GEN = N_GEN + 1
        #print("Generation: ", N_GEN)
    return None, MAX_GEN

def main():
    global N_GEN
    N_GEN = 0
    # Problem: Find equation between numbers to solve relation
    numbers_set = createNumberSet()
    solution = random.randint(MIN_SOL, MAX_SOL)
    print("\nFind equation that is closer to number")
    print("\nNumbers Set:" , numbers_set)
    print("Solution:", solution)
    population = createPopulationNumbers(numbers_set)
    #for p in population:
    #    print(p)
    x,y = geneticProgrammingNumbers(population, solution)
    if x == None:
        print("\nNo solution found")
    else:
        print("\nSolution found:", x)
        print("Result:", x.calculate())
        print("Tolerance:", TOL)
    print("N° generations:", y)
    print("\n######################")
    # Problem: Find equation that is more similar to function
    numbers_set = createNumberSet()
    vars = ["x"]
    function = AST(numbers_set, True, vars)
    function.root = Function(Function(Terminal("x"),Terminal("x"),"*"),Function(Terminal("x"),Terminal(2),"-"),"+")
    print("\nFind equation that is closer to function with variables")
    print("\nNumbers Set:" , numbers_set, vars)
    print("Function:", function)
    population = createPopulationNumbers(numbers_set, True, vars)
    #for p in population:
    #    print(p)
    x,y = geneticProgrammingNumbers(population, function)
    if x == None:
        print("\nNo solution found")
    else:
        print("\nSolution found:", x)
        print("Tolerance:", TOL_EQ)
    print("N° generations:", y)

if __name__ == "__main__":
    main()
