import itertools
import random
import time
import string
import matplotlib.pyplot as plt

N = 8                 # Size of sequence, word, size of dashboard (N x N)
P = 10                  # Size of population
T = int((P * 3) /  4)   # Size of tournament
M_rate = 0.2            # Mutation rate
MAX_gen = 100000         # Maximum number of generation

# Variables
N_gen = 0               # Number of generation for result
selected = []           # Selected secret solution
fitmean = []            # Mean of values in fitness for plot
variance = []           # Variance of values in fitness for plot

def createPopulationQueen(N, P):
    population = []
    while len(population) != P:
        x = []
        for j in range(N):
            x.append(random.randint(0,N-1))
        population.append(x)
    return population

def fitnessQueen(p):
    value = 0
    for i in range(N-1):
        temp = 0
        for j in range(i+1,N):
            # Row collision
            if p[i] == p[j]:
                temp += 1
            # Diagonal collision
            else:
                difcol = abs(i-j) # Difference in columns
                difrow = abs(p[i]-p[j]) # Difference in row
                if difcol == difrow: # Same diagonal for i,j
                    temp += 1
        value += temp # Number of collisions
    return value

def tournamentSelectionQueen(population, fitness):
    best = None
    for i in range(T):
        index = random.randint(0,P-1)
        if (best == None or fitness[index] < fitness[best]):
            best = index
    return population[best]

def reproductionQueen(parent1, parent2):
    new_population = []
    while len(new_population) != P:
        mixpoint = random.randint(0,N-1)
        child = parent1[0:mixpoint] + parent2[mixpoint:]
        for i in range(len(child)):
            if random.random() > M_rate:
                child[i] = random.randint(0,N-1)
        new_population.append(child)
    return new_population

def geneticAlgorithmQueen(population):
    global N_gen
    global fitmean
    global variance
    while N_gen < MAX_gen:
        fitness = []
        for i in range(P):
            f = fitnessQueen(population[i])
            if f == 0:
                return population[i], N_gen
            else:
                fitness.append(f)
        #print("\nFitness:", fitness)
        #print("\n#############################")
        mean = sum(fitness) / len(fitness)
        fitmean.append(mean)
        variance.append(sum((fit - mean) ** 2 for fit in fitness) / len(fitness))
        parent1 = tournamentSelectionQueen(population, fitness)
        parent2 = tournamentSelectionQueen(population, fitness)
        #print("\nTournament parents:", parent1, parent2)
        population = reproductionQueen(parent1, parent2)
        #print("\nNew population:", population)
        N_gen = N_gen + 1
    return "No solution, maximum number of generations reached", MAX_gen

def createPopulationBits(N, P):
    population = []
    while len(population) != P:
        x = []
        for j in range(N):
            x.append(random.randint(0,1))
        population.append(x)
    return population

def createSecretBits(N):
    selected = []
    for i in range(N):
        selected.append(random.randint(0,1))
    return selected

def fitnessBits(indiv):
    value = 0
    for i in range(N):
        if indiv[i] == selected[i]:
            value += 1
    return value

def tournamentSelectionBits(population, fitness):
    best = None
    for i in range(T):
        index = random.randint(0,P-1)
        if (best == None or fitness[index] > fitness[best]):
            best = index
    return population[best]

def reproductionBits(parent1, parent2):
    new_population = []
    while len(new_population) != P:
        mixpoint = random.randint(0,N-1)
        child = parent1[0:mixpoint] + parent2[mixpoint:]
        for i in range(len(child)):
            if random.random() > M_rate:
                child[i] = random.randint(0,1)
        new_population.append(child)
    return new_population

def geneticAlgorithmBits(population):
    global N_gen
    while N_gen < MAX_gen:
        fitness = []
        for i in range(P):
            f = fitnessBits(population[i])
            if f == N:
                return population[i], N_gen
            else:
                fitness.append(f)
        #print("\nFitness:", fitness)
        #print("\n#############################")
        mean = sum(fitness) / len(fitness)
        fitmean.append(mean)
        variance.append(sum((fit - mean) ** 2 for fit in fitness) / len(fitness))
        parent1 = tournamentSelectionBits(population, fitness)
        parent2 = tournamentSelectionBits(population, fitness)
        #print("\nTournament parents:", parent1, parent2)
        population = reproductionBits(parent1, parent2)
        #print("\nNew population:", population)
        N_gen = N_gen + 1
    return "No solution, maximum number of generations reached", MAX_gen

def createPopulationWord(N, P):
    population = []
    while len(population) != P:
        x = []
        for j in range(N):
            x.append(random.choice(string.ascii_lowercase))
        population.append(x)
    return population

def createSecretWord(N):
    selected = []
    for i in range(N):
        selected.append(random.choice(string.ascii_lowercase))
    return selected

def fitnessWord(indiv):
    value = 0
    for i in range(N):
        if indiv[i] == selected[i]:
            value += 1
    return value

def tournamentSelectionWord(population, fitness):
    best = None
    for i in range(T):
        index = random.randint(0,P-1)
        if (best == None or fitness[index] > fitness[best]):
            best = index
    return population[best]

def reproductionWord(parent1, parent2):
    new_population = []
    while len(new_population) != P:
        mixpoint = random.randint(0,N-1)
        child = parent1[0:mixpoint] + parent2[mixpoint:]
        for i in range(len(child)):
            if random.random() > M_rate:
                child[i] = random.choice(string.ascii_lowercase)
        new_population.append(child)
    return new_population

def geneticAlgorithmWord(population):
    global N_gen
    while N_gen < MAX_gen:
        fitness = []
        for i in range(P):
            f = fitnessBits(population[i])
            if f == N:
                return population[i], N_gen
            else:
                fitness.append(f)
        #print("\nFitness:", fitness)
        #print("\n#############################")
        mean = sum(fitness) / len(fitness)
        fitmean.append(mean)
        variance.append(sum((fit - mean) ** 2 for fit in fitness) / len(fitness))
        parent1 = tournamentSelectionWord(population, fitness)
        parent2 = tournamentSelectionWord(population, fitness)
        #print("\nTournament parents:", parent1, parent2)
        population = reproductionWord(parent1, parent2)
        #print("\nNew population:", population)
        N_gen = N_gen + 1
    return "No solution, maximum number of generations reached", MAX_gen

def main():
    global selected
    global N_gen
    global fitmean
    global variance
    # N-Queen
    N_gen = 0
    print("\n##### N-Queen: Solución con algoritmo genético #####")
    population = createPopulationQueen(N, P)
    print("\nPoblación inicial:",population)
    time_start = time.time()
    solution, n = geneticAlgorithmQueen(population)
    time_end = time.time()
    sol_time = time_end - time_start
    print("\n=============================")
    print("\nSolución:", solution)
    print("Tamaño del tablero:", N)
    print("Tamaño de población:", P)
    print("Tamaño del torneo:", T)
    print("Mutation rate:", M_rate)
    print("N° de generaciones:", n)
    print("Tiempo total:", sol_time)
    print("\n=============================")
    print("\nGráfico de fitness promedio y varianza vs N° de generaciones")
    cont = 0
    while cont > 0:
        N_gen = 0
        fitmean = []
        variance = []
        solution, n = geneticAlgorithmQueen(population)
        if n < MAX_gen and n != 0:
            fig = plt.figure()
            plt.plot(range(N_gen), fitmean)
            plt.plot(range(N_gen), variance)
            fig.suptitle('Métricas de desempeño (Valor esperado f=0)', fontsize=15)
            plt.legend(['Fitness promedio', 'Varianza'], loc='upper left')
            plt.xlabel('Generaciones', fontsize=10)
            plt.ylabel('Métricas', fontsize=10)
            plt.show()
            fig.savefig('Nqueen'+str(cont)+'-Gen'+str(n)+'.jpg')
            cont -= 1
    print("\n=============================")
    # Secuencia de Bits secreta
    '''
    N_gen = 0
    fitmean = []
    variance = []
    print("\n##### Secuencia de bits secreta: Solución con algoritmo genético #####")
    population = createPopulationBits(N, P)
    selected = createSecretBits(N)
    print("\nSecuencia secreta elegida:",selected)
    print("\nPoblación inicial:",population)
    time_start = time.time()
    solution, n = geneticAlgorithmBits(population)
    time_end = time.time()
    sol_time = time_end - time_start
    print("\n=============================")
    print("\nSolución:", solution)
    print("Tamaño de la secuencia:", N)
    print("Tamaño de población:", P)
    print("Tamaño del torneo:", T)
    print("Mutation rate:", M_rate)
    print("N° de generaciones:", n)
    print("Tiempo total:", sol_time)
    print("\n=============================")
    print("\nGráfico de fitness promedio y varianza vs N° de generaciones")
    cont = 0
    while cont > 0:
        N_gen = 0
        fitmean = []
        variance = []
        solution, n = geneticAlgorithmBits(population)
        if n < MAX_gen and n != 0:
            fig = plt.figure()
            plt.plot(range(N_gen), fitmean)
            plt.plot(range(N_gen), variance)
            fig.suptitle('Métricas de desempeño (Valor esperado f='+str(N)+')', fontsize=15)
            plt.legend(['Fitness promedio', 'Varianza'], loc='upper left')
            plt.xlabel('Generaciones', fontsize=10)
            plt.ylabel('Métricas', fontsize=10)
            plt.show()
            fig.savefig('Bits'+str(cont)+'-Gen'+str(n)+'.jpg')
            cont -= 1
    print("\n=============================")
    # Palabra secreta
    N_gen = 0
    fitmean = []
    variance = []
    print("\n##### Palabra secreta: Solución con algoritmo genético #####")
    population = createPopulationWord(N, P)
    selected = createSecretWord(N)
    print("\nPalabra secreta elegida:",selected)
    print("\nPoblación inicial:",population)
    time_start = time.time()
    solution, n = geneticAlgorithmWord(population)
    time_end = time.time()
    sol_time = time_end - time_start
    print("\n=============================")
    print("\nSolución:", solution)
    print("Tamaño de la secuencia:", N)
    print("Tamaño de población:", P)
    print("Tamaño del torneo:", T)
    print("Mutation rate:", M_rate)
    print("N° de generaciones:", n)
    print("Tiempo total:", sol_time)
    print("\n=============================")
    print("\nGráfico de fitness promedio y varianza vs N° de generaciones")
    cont = 0
    while cont > 0:
        N_gen = 0
        fitmean = []
        variance = []
        solution, n = geneticAlgorithmWord(population)
        if n < MAX_gen and n != 0:
            fig = plt.figure()
            plt.plot(range(N_gen), fitmean)
            plt.plot(range(N_gen), variance)
            fig.suptitle('Métricas de desempeño (Valor esperado f='+str(N)+')', fontsize=15)
            plt.legend(['Fitness promedio', 'Varianza'], loc='upper left')
            plt.xlabel('Generaciones', fontsize=10)
            plt.ylabel('Métricas', fontsize=10)
            plt.show()
            fig.savefig('Secretword'+str(cont)+'-Gen'+str(n)+'.jpg')
            cont -= 1
    print("\n=============================")
    '''
if __name__ == '__main__':
    main()
