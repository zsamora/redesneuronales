Para ejecutar el programa, basta con ingresar el siguiente comando en consola:

>$ python3 geneticalgorithm.py

Esto ejecutará el algoritmo genético para resolver el problema de N-queen con
los valores globales definidos al inicio del código. Las variables (para que
puedan modificarse fácilmente son:

  - N = Tamaño del tablero (N x N), correspondiente a la codificación
        recomendada del tablero de una lista de largo N, tal que cada elemento
        de la lista corresponde a la posición de la reina en esa columna
  - P = Tamaño de la población (inicial como de las posteriores)
  - T = Tamaño del torneo (se utiliza un valor recomendado de 3/4 * P)
  - M_rate = Mutation rate, para realizar la mutación en los hijos
  - MAX_gen = Máximo número de generaciones para evitar que el algoritmo
              corra infinitamente
