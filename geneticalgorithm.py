import itertools
import random
import time

N = 4                # Size of bits, size of dashboard (N x N)
P = 4                # Size of population
T = int((P * 3) /  4) # Size of tournament
M_rate = 0.2          # Mutation rate
MAX_gen = 10000       # Maximum number of generation

# Variables
N_gen = 0             # Number of generation for result

# Create population for N-Queen (List of list with codification [X Y] with X
# the position of the first queen in the first column and Y the second one
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
    # Row collision
    for i in range(N-1):
        temp = 0
        for j in range(i+1,N):
            if p[i] == p[j]:
                temp += 1
        value += temp
    # Diagonal collision
    for i in range(N-1):
        temp = 0
        for j in range(i+1,N):
            difcol = abs(i-j) # Difference in columns
            difrow = abs(p[i]-p[j]) # Difference in row
            if difcol == difrow: # Same diagonal for i,j
                temp += 1
        value += temp # Number of collisions
    return value

def tournamentSelectionQueen(population, fitness):
    best = None
    for i in range(T):
        index = random.randint(0,N-1)
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
    while N_gen < MAX_gen:
        fitness = []
        for i in range(P):
            f = fitnessQueen(population[i])
            if f == 0:
                return population[i], N_gen
            else:
                fitness.append(f)
        print("\nFitness:", fitness)
        print("\n#############################")
        parent1 = tournamentSelectionQueen(population, fitness)
        parent2 = tournamentSelectionQueen(population, fitness)
        print("\nTournament parents:", parent1, parent2)
        population = reproductionQueen(parent1, parent2)
        print("\nNew population:", population)
        N_gen = N_gen + 1
    return "No solution, maximum number of generations reached", MAX_gen

def main():
    global selected
    population = createPopulationQueen(N, P)
    print("\nInitial Population:",population)
    time_start = time.time()
    solution, n = geneticAlgorithmQueen(population)
    time_end = time.time()
    sol_time = time_end - time_start
    print("\n=============================")
    print("\nSolution:", solution)
    print("Tamaño del tablero:", N)
    print("Tamaño de población:", P)
    print("Tamaño del torneo:", T)
    print("N° de generaciones:", n)
    print("Tiempo total:", sol_time)

if __name__ == '__main__':
    main()
